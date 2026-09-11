"""Task9: production UI mixins, instance stores and fake-SDK voltage capture."""
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtWidgets import QPushButton

from base.data_struct.data_deal_struct import DataDealStruct
from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingResult
from base.recording_result_reader import RecordingAudio
from base.ve3668n_capture import Ve3668nInputStream
from base.ve3668n_stores import VECalibrationStore, VEInputProfileStore
from base.wav_calibration_metadata import inspect_wav_calibration_metadata
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from ui.sequence import sequence_widget_analysis_ops as analysis
from ui.sequence import sequence_widget_streaming_ops as streaming
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from unit_test.base.ve3668n_fakes import CaptureSDK, device_info, input_config
from unit_test.ui.test_streaming_event_dispatch import _WorkflowHost


class RecordingBridgeFake:
    """Inject lifecycle delivery, leaving request/arrays/plots as real values."""
    def __init__(self):
        self.service = SimpleNamespace(busy=False, is_path_leased=lambda path: True)
        self.requests = []

    def start(self, request, callbacks):
        self.requests.append(request)
        self.callbacks = callbacks
        return SimpleNamespace(
            request=request, generation=1, released=threading.Event(),
            release_error=None, state="recording", worker_pid=123,
            accept_result=mock.Mock(), reject_result=mock.Mock(), cancel=mock.Mock(),
        )


class RecordingHost(_WorkflowHost, SequenceWidgetUiOpsMixin):
    replayer_btn = data_btn = lineedit_type = lineedit_s_or_n = barcode_scanner_box = None
    # Exercise the actual main workflow instead of the older fixture's shortcuts.
    reset_work_pram = analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram
    checked_work_status_message = analysis.SequenceWidgetAnalysisOpsMixin.checked_work_status_message
    _capture_recording_wav_calibration_metadata = analysis.SequenceWidgetAnalysisOpsMixin._capture_recording_wav_calibration_metadata
    _should_use_streaming_recording = analysis.SequenceWidgetAnalysisOpsMixin._should_use_streaming_recording
    _on_streaming_complete = streaming.SequenceWidgetStreamingOpsMixin._on_streaming_complete
    _cleanup_streaming_resources = streaming.SequenceWidgetStreamingOpsMixin._cleanup_streaming_resources


@pytest.fixture
def host_factory(ui_qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(analysis.QMessageBox, "warning", mock.Mock())
    monkeypatch.setattr(analysis, "get_recorded_info", lambda *a, **k: (
        str(tmp_path / "main.wav"), {"name": "capture", "labels": "not_labeled"}))
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(streaming, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    hosts = []

    def make(rate=51200, live=False, *, host_type=RecordingHost):
        host = host_type({}, [])
        host.recording_bridge = RecordingBridgeFake()
        host.mic = device_info(input_config=input_config(rate))
        host.speaker = None  # VE input never requires an output device.
        host.mic_channels = [7, 1]
        host._recording_input_channels = None
        host._active_input_channels = []
        host.channel_workspace = ChannelPlotWorkspace()
        host.ve_profile_store = VEInputProfileStore(tmp_path / "profiles.json")
        host.ve_calibration_store = VECalibrationStore(tmp_path / "calibrations.json")
        host.ve_profile_store.set_sample_rate(host.mic, rate, host.ve_calibration_store)
        # Use the real clear_data method without mutating the legacy singleton.
        host.data_struct = object.__new__(DataDealStruct)
        host.data_struct.sample_rate = 32000
        host.data_struct.clear_data()
        detail = dict(total_time=.04, startup_trim_ms=10, use_streaming_recording=live,
                      audio_validation={"enabled": False})
        host.sequence_config = [{"seq1": {"acq": {"detail": detail}}}]
        host.lineedit_type = SimpleNamespace(text=lambda: "test")
        host.lineedit_s_or_n = SimpleNamespace(text=lambda: "serial")
        host._current_run_recording_token = "test"
        host._resolve_recording_name_suffix = lambda: ""
        host._get_active_product_condition_key = lambda: ""
        host._resolve_active_recording_waveform_direction = lambda fallback="": ""
        host._begin_recent_session_for_current_run = mock.Mock()
        host._discard_current_recent_session = mock.Mock()
        host._cache_condition_record = mock.Mock()
        host._update_current_recent_session_result = mock.Mock()
        host._is_manual_product_condition_cycle_active = lambda: False
        # This fixture covers capture; original subprocess routing is tested separately.
        host._should_run_silent_analysis_after_recording = lambda: False
        host.run = mock.Mock(return_value=True)
        host.count_board = SimpleNamespace(mode="view")
        host.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
        host.data_btn = QPushButton()
        host.replayer_btn = QPushButton()
        host.update_player_btn_is_paused = mock.Mock()
        host._send_recording_tcp_finish = mock.Mock()
        host.saved = save
        host.refresh_channel_windows()
        hosts.append(host)
        return host

    yield make
    for host in hosts:
        # Invalidate queued release-check timers before destroying Qt widgets.
        host._recording_process_id = None
        host.channel_workspace.close()
        host.data_btn.close()
        host.replayer_btn.close()


def capture_audio(request):
    capture = RecordingCapture(request, ve_stream_factory=lambda **kwargs: Ve3668nInputStream(
        **kwargs, sdk_factory=CaptureSDK))
    capture.start()
    descriptor = capture.wait(5)
    assert isinstance(descriptor, RecordingResult), descriptor
    multi, rate = sf.read(request.path, dtype="float32", always_2d=True)
    assert rate == request.sample_rate
    return capture, RecordingAudio(descriptor, multi, multi.mean(axis=1))


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("live", [False, True])
def test_effective_rate_is_resolved_before_target_trim_and_timeline(host_factory, rate, live):
    host = host_factory(rate, live)
    recorded, actual_rate = host.reset_work_pram("not_labeled")
    assert actual_rate == rate
    assert host.data_struct.sample_rate == recorded["sample_rate"] == recorded["sr"] == rate
    assert recorded["num_frames"] == int(.04 * rate) + round(.01 * rate)
    assert recorded["startup_trim_samples"] == round(.01 * rate)
    # Isolate the rate contract before the next TDD cycle wires the UI builder.
    host._recording_wav_calibration_metadata = build_recording_wav_calibration_metadata(
        host._recording_input_channels, recorded["device"], ve_calibration_store=host.ve_calibration_store)
    host._start_process_recording(recorded, actual_rate)
    request = host._recording_process_session.request
    assert request.sample_rate == rate
    assert request.target_samples == recorded["num_frames"]
    assert request.trim_samples == round(.01 * rate)
    assert request.effective_streaming is live
    capture, audio = capture_audio(request)
    assert audio.descriptor.raw_frames == request.target_samples
    assert audio.descriptor.final_frames == int(.04 * rate)
    info = sf.info(request.path)
    assert info.samplerate == rate and info.frames == int(.04 * rate)
    preview = capture.snapshot(generation=1, sequence=1)
    if live:
        assert preview is not None
        assert preview.time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        assert preview.waveforms[0].time[-1] == 0.0
    else:
        assert preview is None
    time_axis, _ = host._prepare_waveform_display_data(audio.multi[:, 0], rate)
    assert time_axis[-1] == pytest.approx((len(audio.multi) - 1) / rate)


def test_ve_request_and_capture_use_frozen_cumulative_preview_mode(host_factory):
    host = host_factory(48_000, True)
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = PREVIEW_TIME_MODE_CUMULATIVE
    recorded, rate = host.reset_work_pram("not_labeled")
    host._recording_wav_calibration_metadata = build_recording_wav_calibration_metadata(
        host._recording_input_channels,
        recorded["device"],
        ve_calibration_store=host.ve_calibration_store,
    )

    host._start_process_recording(recorded, rate)
    request = host._recording_process_session.request
    detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = PREVIEW_TIME_MODE_RELATIVE_LATEST
    capture, _audio = capture_audio(request)
    preview = capture.snapshot(generation=1, sequence=1)

    assert request.preview_time_mode == PREVIEW_TIME_MODE_CUMULATIVE
    assert preview.time_mode == PREVIEW_TIME_MODE_CUMULATIVE
    assert preview.waveforms[0].time[0] == 0.0
    assert preview.waveforms[0].time[-1] > 0.0


@pytest.mark.parametrize("raw_rate", [48000.0, "48000", True, False, None, 96000])
def test_start_does_not_coerce_invalid_ve_rate(host_factory, raw_rate):
    host = host_factory(48000)
    recorded, _ = host.reset_work_pram("not_labeled")
    host._recording_wav_calibration_metadata = build_recording_wav_calibration_metadata(
        host._recording_input_channels, recorded["device"], ve_calibration_store=host.ve_calibration_store)
    with pytest.raises((TypeError, ValueError)):
        host._start_process_recording(recorded, raw_rate)
    assert host.recording_bridge.requests == []


def test_ve_channel_order_survives_refresh_request_metadata_and_wav(host_factory):
    host = host_factory()
    assert host._configured_input_channels == (7, 1)
    assert [window.channel_index for window in host.channel_workspace.all_subwindows()] == [7, 1]
    recorded, rate = host.reset_work_pram("not_labeled")
    assert recorded["input_channels"] == [7, 1]
    host._recording_wav_calibration_metadata = build_recording_wav_calibration_metadata(
        host._recording_input_channels, recorded["device"], ve_calibration_store=host.ve_calibration_store)
    host._start_process_recording(recorded, rate)
    request = host._recording_process_session.request
    assert request.channels == (7, 1)
    _, audio = capture_audio(request)
    np.testing.assert_array_equal(audio.multi, np.tile(np.float32([8.25, 2.5]), (len(audio.multi), 1)))
    metadata = inspect_wav_calibration_metadata(request.path).metadata
    assert [entry["physical_input_channel"] for entry in metadata["recorded_channels"]] == [7, 1]


@pytest.mark.parametrize("channels", [[], [7, 7], [True, 1], [7.0, 1], [8], [2]])
def test_ve_channel_refresh_rejects_invalid_order_without_repair(host_factory, channels):
    host = host_factory()
    previous = tuple(host._configured_input_channels)
    host.mic_channels = channels
    host.refresh_channel_windows()
    assert host._channel_selection_error
    assert host._configured_input_channels == previous
    with pytest.raises(ValueError):
        host.reset_work_pram("not_labeled")


def save_calibration(host, physical=7, factor=10):
    host.ve_calibration_store.save(
        host.mic, physical, v2pa_factor=factor, standard_spl=94,
        calibration_sample_rate=51200, calibration_duration_seconds=10,
        calibrated_at="2026-08-28T10:00:00+08:00")


@pytest.mark.parametrize("measured", [False, True])
def test_request_freezes_shared_store_profile_and_calibration(host_factory, monkeypatch, measured):
    host = host_factory()
    if measured:
        save_calibration(host)
    # External persisted profile is authoritative, even if the UI copy is stale.
    host.ve_profile_store.set_sample_rate(host.mic, 44100, host.ve_calibration_store)
    forbidden = mock.Mock(side_effect=AssertionError("legacy registry must not supply VE coefficients"))
    monkeypatch.setattr("base.recording_calibration_snapshot.load_mic_channel_calibrations", forbidden)
    host.judge_play_and_record()
    assert len(host.recording_bridge.requests) == 1
    request = host._recording_process_session.request
    original = request.calibration_metadata.to_dict()
    assert request.sample_rate == original["acquisition"]["sample_rate"] == 44100
    assert original["recorded_channels"][0]["factor_source"] == ("measured" if measured else "none")
    assert original["recorded_channels"][0]["v2pa_factor"] == (10 if measured else None)
    assert original["recorded_channels"][1]["factor_source"] == "none"
    if measured:
        assert original["recorded_channels"][0]["calibration"]["sample_rate"] == 51200
    host.ve_profile_store.set_sample_rate(host.mic, 48000, host.ve_calibration_store)
    save_calibration(host, factor=99)
    save_calibration(host, physical=1, factor=42)
    host.mic["machine_id"] = "changed-device"
    host.mic_channels.reverse()
    host.data_struct.sample_rate = 96000
    _, audio = capture_audio(request)
    assert request.calibration_metadata.to_dict() == original
    assert inspect_wav_calibration_metadata(request.path).metadata == original
    np.testing.assert_array_equal(audio.multi, np.tile(np.float32([8.25, 2.5]), (len(audio.multi), 1)))
    forbidden.assert_not_called()


def test_legacy_rate_coercion_does_not_change_trim_calculation(host_factory):
    from unit_test.base.recording_process_fakes import device_info as soundcard
    from base.play_and_record import resolve_startup_trim_samples
    host = host_factory()
    host.mic = soundcard()
    host.mic_channels = [2, 0, 2]
    host.refresh_channel_windows()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(startup_trim_ms=1000, monitor_playback=True)
    host.speaker = soundcard()
    host.data_struct.sample_rate = 48000.9
    recorded, raw_rate = host.reset_work_pram("not_labeled")
    assert raw_rate == recorded["sr"] == recorded["sample_rate"] == 48000.9
    assert host._recording_input_channels == (0, 2)
    host._start_process_recording(recorded, raw_rate)
    request = host._recording_process_session.request
    assert request.sample_rate == 48000  # legacy request coercion remains
    assert request.trim_samples == resolve_startup_trim_samples(detail, raw_rate)
    assert request.trim_samples == recorded["startup_trim_samples"]
    assert dict(request.monitor) == {}
    assert request.effective_streaming is host._should_use_streaming_recording()


def test_legacy_snapshot_value_error_still_restores_controls_before_raising(host_factory, monkeypatch):
    from unit_test.base.recording_process_fakes import device_info as soundcard
    host = host_factory()
    host.mic = soundcard()
    host.mic_channels = [0, 2]
    host.refresh_channel_windows()
    monkeypatch.setattr(analysis, "build_recording_wav_calibration_metadata",
                        mock.Mock(side_effect=ValueError("programming error")))
    with pytest.raises(ValueError, match="programming error"):
        host.judge_play_and_record()
    assert not host._record_workflow_busy and not host.player_status_flag
    assert host.data_btn.isEnabled() and host.replayer_btn.isEnabled()


def test_busy_and_cancel_restore_channel_selection_without_new_request(host_factory):
    host = host_factory()
    host.recording_bridge.service.busy = True
    host.judge_play_and_record()
    assert host.recording_bridge.requests == []
    host.recording_bridge.service.busy = False
    host.judge_play_and_record()
    session = host._recording_process_session
    assert host.player_status_flag and host._record_workflow_busy
    assert not host.data_btn.isEnabled() and not host.replayer_btn.isEnabled()
    host.mic_channels = [1, 7]
    host.refresh_channel_windows()
    assert host._recording_input_channels == (7, 1)
    assert host._pending_configured_input_channels == (1, 7)
    host.judge_play_and_record()
    assert len(host.recording_bridge.requests) == 1
    host._on_process_recording_cancelled(session, None)
    assert host._recording_input_channels is None
    assert host._active_input_channels == [1, 7]
    assert host._recording_ve_device is None
    assert host._recording_wav_calibration_metadata is None
    assert host.data_btn.isEnabled() and host.replayer_btn.isEnabled()
    assert not host.player_status_flag and not host._record_workflow_busy
    host.run.assert_not_called()


def test_imported_file_keeps_rate_outside_ve_whitelist(host_factory, tmp_path):
    host = host_factory()
    host.sequence_config[0]["seq1"]["acq"]["mode"] = "IMPORT_AUDIO"
    path = tmp_path / "historical.wav"
    samples = np.tile(np.float32([8.25, 2.5]), (320, 1))
    sf.write(path, samples, 96000, subtype="FLOAT")
    host._load_audio_file_to_data_struct(str(path), sample_rate=sf.info(path).samplerate)
    assert host.data_struct.sample_rate == 96000
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, samples)


@pytest.mark.parametrize("failure", ["unavailable", "profile_io", "profile_type", "calibration_io", "snapshot_invalid"])
def test_invalid_ve_initialization_is_actionable_and_restores_controls(host_factory, monkeypatch, failure):
    host = host_factory()
    if failure == "unavailable":
        host.mic["available"] = False
    elif failure == "profile_io":
        monkeypatch.setattr(host.ve_profile_store, "load", mock.Mock(side_effect=OSError("profile denied")))
    elif failure == "profile_type":
        monkeypatch.setattr(host.ve_profile_store, "load", lambda *a: input_config(48000.0))
    elif failure == "calibration_io":
        # First observation belongs to profile loading, second to the WAV builder.
        monkeypatch.setattr(host.ve_calibration_store, "observe", mock.Mock(side_effect=[{}, OSError("calibration denied")]))
    else:
        monkeypatch.setattr(analysis, "build_recording_wav_calibration_metadata", lambda *a, **k: {"backend": "vkinging"})
    host.judge_play_and_record()
    assert host.recording_bridge.requests == []
    assert not host.player_status_flag and not host._record_workflow_busy
    assert host.data_btn.isEnabled() and host.replayer_btn.isEnabled()
    assert host._recording_input_channels is None
    assert getattr(host, "_recording_ve_device", None) is None
    assert host._recording_wav_calibration_metadata is None
    warning = analysis.QMessageBox.warning.call_args.args[-1]
    assert warning


def started_audio(host):
    host.judge_play_and_record(tcp_completion_address=("127.0.0.1", 1234))
    session = host._recording_process_session
    capture, audio = capture_audio(session.request)
    return session, capture, audio


def finish_ve_capture(host, session, audio, outcome="success"):
    if outcome == "success":
        host._on_process_recording_result(session, audio)
        session.accept_result.assert_called_once()
        session.state = "completed"
        host._on_process_recording_accepted(session, audio)
        session.released.set()
        host._on_process_recording_released(session)
        np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
    elif outcome == "cancel":
        host._on_process_recording_cancelled(session, None)
    else:
        host._on_process_recording_failed(session, SimpleNamespace(stage="read", message="failed"))
    assert not host.player_status_flag and not host._record_workflow_busy
    assert host._recording_input_channels is None


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
@pytest.mark.parametrize("configured_product_rate", [None, 32000])
def test_soundcard_admission_after_ve_uses_product_rate(
        host_factory, rate, live, outcome, configured_product_rate):
    from unit_test.base.recording_process_fakes import device_info as soundcard
    host = host_factory(rate, live)
    ve_rate = configured_product_rate if configured_product_rate is not None else rate
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    if configured_product_rate is not None:
        detail["sample_rate"] = configured_product_rate
    host.mic_channels = [1]
    host.refresh_channel_windows()
    ve_device = host.mic
    session, _, audio = started_audio(host)
    finish_ve_capture(host, session, audio, outcome)
    window = host.channel_workspace.all_subwindows()[0]
    host.mic = soundcard()
    host.refresh_channel_windows()
    assert host.channel_workspace.all_subwindows()[0] is window
    # Hardware selection must not reinterpret the previous VE audio/timebase.
    assert host.data_struct.sample_rate == ve_rate
    assert sf.info(session.request.path).samplerate == ve_rate
    if outcome == "success":
        np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
        assert window.plot_item.getData()[0][-1] == pytest.approx((len(audio.multi) - 1) / ve_rate)
    recorded, actual_rate = host.reset_work_pram("not_labeled")
    assert actual_rate == recorded["sample_rate"] == recorded["sr"] == 32000
    assert host.data_struct.sample_rate == 32000
    assert recorded["num_frames"] == 1600
    assert recorded["startup_trim_samples"] == 320
    host._recording_wav_calibration_metadata = None
    host._start_process_recording(recorded, actual_rate)
    request = host._recording_process_session.request
    assert request.sample_rate == 32000 and request.target_samples == 1600 and request.trim_samples == 320
    assert request.effective_streaming is live
    host._on_process_recording_cancelled(host._recording_process_session, None)
    host.mic = ve_device
    host.refresh_channel_windows()
    recorded, actual_rate = host.reset_work_pram("not_labeled")
    assert actual_rate == recorded["sr"] == ve_rate
    assert recorded["num_frames"] == int(.04 * ve_rate) + round(.01 * ve_rate)
    assert recorded["startup_trim_samples"] == round(.01 * ve_rate)


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
def test_current_product_rate_change_during_ve_wins_at_next_soundcard_admission(
        host_factory, rate, live, outcome):
    from unit_test.base.recording_process_fakes import device_info as soundcard
    host = host_factory(rate, live)
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail["sample_rate"] = 32000
    host.init_data_struct_stimulus_config()
    host.mic_channels = [1]
    host.refresh_channel_windows()
    session, _, audio = started_audio(host)
    # A product configuration change cannot modify the admitted VE request.
    detail["sample_rate"] = 22050
    host.init_data_struct_stimulus_config()
    assert session.request.sample_rate == 32000
    finish_ve_capture(host, session, audio, outcome)
    host.mic = soundcard()
    host.refresh_channel_windows()
    recorded, actual_rate = host.reset_work_pram("not_labeled")
    assert actual_rate == recorded["sr"] == recorded["sample_rate"] == 22050
    assert recorded["num_frames"] == int(.04 * 22050) + round(.01 * 22050)
    assert recorded["startup_trim_samples"] == round(.01 * 22050)


@pytest.mark.parametrize("after_ve", [False, True])
def test_soundcard_admission_uses_product_rate_not_imported_file_rate(host_factory, tmp_path, after_ve):
    from unit_test.base.recording_process_fakes import device_info as soundcard
    host = host_factory()
    acq = host.sequence_config[0]["seq1"]["acq"]
    acq["detail"]["sample_rate"] = 32000
    if after_ve:
        session, _, audio = started_audio(host)
        finish_ve_capture(host, session, audio)
    acq["mode"] = "IMPORT_AUDIO"
    path = tmp_path / "imported.wav"
    samples = np.full((320, 2), .05, dtype=np.float32)
    sf.write(path, samples, 96000, subtype="FLOAT")
    host._load_audio_file_to_data_struct(str(path), sample_rate=96000)
    host.mic = soundcard()
    host.mic_channels = [0, 1]
    host.refresh_channel_windows()
    assert host.data_struct.sample_rate == 96000
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, samples)
    acq.pop("mode")
    recorded, actual_rate = host.reset_work_pram("not_labeled")
    assert actual_rate == recorded["sr"] == 32000
    assert recorded["num_frames"] == 1600 and recorded["startup_trim_samples"] == 320


@pytest.mark.parametrize("fault", [
    "request_id", "purpose", "path", "channels", "sample_rate", "rate_type",
    "raw_frames", "final_frames", "count_type", "metadata_appended", "handles_released",
    "shape", "nan", "mono", "context", "dtype",
])
def test_ve_invalid_descriptor_is_rejected_before_accept_or_analysis(host_factory, fault):
    host = host_factory()
    session, _, audio = started_audio(host)
    descriptor = audio.descriptor
    changes = {
        "request_id": {"request_id": "other"}, "purpose": {"purpose": "calibration"},
        "path": {"path": descriptor.path + ".other"}, "channels": {"channels": (1, 7)},
        "sample_rate": {"sample_rate": 48000}, "rate_type": {"sample_rate": 51200.0},
        "raw_frames": {"raw_frames": descriptor.raw_frames + 1},
        "final_frames": {"final_frames": descriptor.final_frames + 1},
        "count_type": {"final_frames": float(descriptor.final_frames)},
        "metadata_appended": {"metadata_appended": False},
        "handles_released": {"handles_released": False},
    }
    if fault in changes:
        audio = replace(audio, descriptor=replace(descriptor, **changes[fault]))
    elif fault == "shape":
        audio = replace(audio, multi=audio.multi[:, :1])
    elif fault == "nan":
        multi = audio.multi.copy()
        multi[0, 0] = np.nan
        audio = replace(audio, multi=multi, mono=multi.mean(axis=1))
    elif fault == "mono":
        audio = replace(audio, mono=audio.mono + 1)
    elif fault == "dtype":
        audio = replace(audio, multi=audio.multi.astype(np.float64), mono=audio.mono.astype(np.float64))
    else:
        metadata = session.request.calibration_metadata.to_dict()
        metadata["acquisition"]["machine_id"] = "other-device"
        session.request = SimpleNamespace(**{
            field: getattr(session.request, field) for field in session.request.__dataclass_fields__
        })
        session.request.calibration_metadata = metadata
    host._on_process_recording_result(session, audio)
    session.reject_result.assert_called_once()
    session.accept_result.assert_not_called()
    # A stale/misordered accepted/released event must not resurrect rejected data.
    session.state = "completed"
    session.released.set()
    host._on_process_recording_accepted(session, audio)
    host._on_process_recording_released(session)
    assert host.data_struct.store_wave_data_multi is None
    host.run.assert_not_called()
    host.saved.assert_not_called()
    host._send_recording_tcp_finish.assert_not_called()


@pytest.mark.parametrize("failure", ["not_released", "release_error", "release_timeout", "late_failure", "cancel", "changed_request"])
def test_ve_unreleased_or_failed_accepted_result_never_publishes(host_factory, failure):
    host = host_factory()
    session, _, audio = started_audio(host)
    host._on_process_recording_result(session, audio)
    session.accept_result.assert_called_once()
    session.state = "completed"
    host._on_process_recording_accepted(session, audio)
    if failure == "release_error":
        session.release_error = "close denied"
        host._on_process_recording_release_failed(session, session.release_error)
    elif failure == "release_timeout":
        host._recording_release_deadline = 0
        host._check_accepted_recording_release(session)
    elif failure == "late_failure":
        host._on_process_recording_failed(session, SimpleNamespace(stage="release", message="failed"))
        host._on_process_recording_accepted(session, audio)
        session.released.set()
    elif failure == "cancel":
        host._cancel_process_recording()
        host._on_process_recording_accepted(session, audio)
        session.released.set()
    elif failure == "changed_request":
        # A replaced accepted context cannot borrow the original UI validation.
        session.request = replace(session.request, trim_samples=0)
        session.released.set()
    host._publish_process_recording(session)
    assert host.data_struct.store_wave_data_multi is None
    host.saved.assert_not_called()
    host.run.assert_not_called()
    host._send_recording_tcp_finish.assert_not_called()
    if failure in ("release_error", "release_timeout"):
        # A later release cannot convert the already reported failure to success.
        session.released.set()
        host._on_process_recording_accepted(session, audio)
        host._on_process_recording_released(session)
        host.run.assert_not_called()
        assert "不会改变" not in analysis.QMessageBox.warning.call_args.args[-1]


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("live", [False, True])
def test_publish_keeps_frozen_rate_metadata_and_raw_voltage(host_factory, monkeypatch, rate, live):
    host = host_factory(rate, live)
    session, capture, audio = started_audio(host)
    metadata = session.request.calibration_metadata
    forbidden = mock.Mock(side_effect=AssertionError("prefinalized VE must not retrim/rewrite/revalidate"))
    monkeypatch.setattr(streaming, "validate_recorded_audio", forbidden)
    host._rewrite_recorded_wav = forbidden
    host._append_recording_wav_calibration_metadata = forbidden
    host.data_struct.sample_rate = 96000
    save_calibration(host, factor=99)
    host.ve_profile_store.set_sample_rate(host.mic, 48000, host.ve_calibration_store)
    host.mic["machine_id"] = "later-device"
    host._on_process_recording_result(session, audio)
    assert host.data_struct.store_wave_data_multi is None
    session.state = "completed"
    host._on_process_recording_accepted(session, audio)
    assert host.data_struct.store_wave_data_multi is None
    session.released.set()
    host._on_process_recording_released(session)
    assert host.data_struct.sample_rate == rate
    assert host.recorded_signal_info["sample_rate"] == rate
    assert host.data_struct.wav_calibration_metadata == metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, audio.multi)
    assert host.data_btn.isEnabled() and host.replayer_btn.isEnabled()
    assert not host._record_workflow_busy and not host.player_status_flag
    host.run.assert_not_called()
    host.saved.assert_called_once()
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1234))
    forbidden.assert_not_called()
    for column, window in enumerate(host.channel_workspace.all_subwindows()):
        x, y = window.plot_item.getData()
        assert x[-1] == pytest.approx((len(audio.multi) - 1) / rate)
        np.testing.assert_array_equal(y, audio.multi[:, column])
        assert "Amplitude(V)" in window.plot_widget.getAxis("left").labelText
        assert "未校准，仅电压数据" in window.toolTip()
    host._on_process_recording_released(session)
    host.run.assert_not_called()


@pytest.mark.parametrize("measured", [False, True])
def test_preview_is_raw_voltage_and_uses_frozen_calibration_hint(host_factory, measured):
    host = host_factory(live=True)
    if measured:
        save_calibration(host)
    session, capture, audio = started_audio(host)
    preview = capture.snapshot(generation=session.generation, sequence=1)
    save_calibration(host, physical=1, factor=42)
    host._on_process_recording_preview(session, preview)
    for column, window in enumerate(host.channel_workspace.all_subwindows()):
        x, y = window.plot_item.getData()
        np.testing.assert_array_equal(y, preview.waveforms[column].amplitude)
        np.testing.assert_array_equal(x, preview.waveforms[column].time)
        assert x[-1] == 0.0
        assert window.is_live_preview is True
        assert "Amplitude(V)" in window.plot_widget.getAxis("left").labelText
        assert ("实测校准有效" if measured and column == 0 else "未校准，仅电压数据") in window.toolTip()


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("replacement", ["preview", "final"])
@pytest.mark.parametrize("other_hint", ["empty", "prior", "changed"])
def test_legacy_waveform_replacement_resets_only_ve_owned_tooltip(
        host_factory, measured, replacement, other_hint):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    from unit_test.base.recording_process_fakes import device_info as soundcard
    host = host_factory(live=True)
    host.mic_channels = [1]
    host.refresh_channel_windows()
    window = host.channel_workspace.all_subwindows()[0]
    expected_hint = "channel note" if other_hint == "prior" else ""
    window.setToolTip(expected_hint)
    if measured:
        save_calibration(host, physical=1)
    session, capture, audio = started_audio(host)
    # Repeated previews and final publication must retain the original hint.
    for sequence in (1, 2):
        host._on_process_recording_preview(session, capture.snapshot(generation=1, sequence=sequence))
    finish_ve_capture(host, session, audio)
    assert ("实测校准有效" if measured else "未校准，仅电压数据") in window.toolTip()
    if other_hint == "changed":
        expected_hint = "updated channel note"
        window.setToolTip(expected_hint)
    displayed_hint = window.toolTip()
    previous_x, previous_y = window.plot_item.getData()
    host.mic = soundcard()
    host.refresh_channel_windows()
    assert host.channel_workspace.all_subwindows()[0] is window
    assert window.toolTip() == displayed_hint
    np.testing.assert_array_equal(window.plot_item.getData()[0], previous_x)
    np.testing.assert_array_equal(window.plot_item.getData()[1], previous_y)
    recorded, rate = host.reset_work_pram("not_labeled")
    host._recording_wav_calibration_metadata = None
    host._start_process_recording(recorded, rate)
    assert window.toolTip() == displayed_hint  # Admission is not waveform replacement.
    samples = np.full((int(.04 * rate), 1), .05, dtype=np.float32)
    if replacement == "preview":
        session = host._recording_process_session
        waveform = StreamingWaveformSnapshot(
            (np.arange(len(samples), dtype=np.float64) - len(samples) + 1) / rate,
            samples[:, 0], len(samples))
        preview = RecordingPreview(session.request.request_id, session.generation, 1,
                                   len(samples), (1,), (waveform,),
                                   PREVIEW_TIME_MODE_RELATIVE_LATEST)
        host._on_process_recording_preview(session, preview)
        assert host._recording_process_preview_enabled
    else:
        assert host._on_streaming_complete(
            recorded_mono=samples[:, 0], recorded_multi=samples, sample_rate=rate,
            prefinalized=True, final_waveform_windows=(window,)) is True
    np.testing.assert_array_equal(window.plot_item.getData()[1], samples[:, 0])
    expected_endpoint = 0.0 if replacement == "preview" else (len(samples) - 1) / rate
    assert window.plot_item.getData()[0][-1] == pytest.approx(expected_endpoint)
    assert window.is_live_preview is (replacement == "preview")
    assert window.toolTip() == expected_hint




@pytest.mark.parametrize("outcome", ["ordinary", "streaming", "cancel", "close", "metadata_false", "release_error"])
def test_real_service_bridge_ve_lifecycle(host_factory, ui_qapp, tmp_path, outcome):
    from base.recording_result_reader import ResultReader
    from base.recording_service import RecordingService
    from ui.recording_service_bridge import RecordingServiceBridge
    from unit_test.ui.test_recording_process_integration import pump

    host = host_factory(live=outcome == "streaming")
    # The native owner/writer/reader/service/Qt delivery are production code.
    # Only the external SDK and business analysis/DB are replaced.
    service = RecordingService(
        backend_factory="unit_test.base.ve3668n_fakes:capture_dependencies",
        backend_options={"trace_path": str(tmp_path / "sdk.jsonl"), "read_delay": .03},
        preview_interval=.01)
    if outcome == "metadata_false":
        service._reader_factory = lambda descriptor, completed: ResultReader(
            replace(descriptor, metadata_appended=False), completed)
    bridge = RecordingServiceBridge(service)
    host.recording_bridge = bridge
    if outcome in ("cancel", "close", "streaming"):
        host.sequence_config[0]["seq1"]["acq"]["detail"]["total_time"] = .5
    previews = []
    on_preview = host._on_process_recording_preview

    def preview(session, value):
        previews.append(value)
        on_preview(session, value)

    host._on_process_recording_preview = preview
    session = None
    try:
        host.judge_play_and_record(tcp_completion_address=("127.0.0.1", 1234))
        session = host._recording_process_session
        assert host.player_status_flag and not host.data_btn.isEnabled()
        if outcome == "release_error":
            def deny_cleanup(path):
                raise OSError("test lease cleanup denied")
            assert service.defer_path_cleanup(session.request.path, deny_cleanup)
        if outcome in ("cancel", "close"):
            pump(ui_qapp, lambda: session.state == "recording")
            if outcome == "cancel":
                session.cancel()
            else:
                host._cancel_process_recording()
                bridge.shutdown()
        if outcome == "release_error":
            pump(ui_qapp, lambda: session.release_error is not None and not host._record_workflow_busy)
            assert service.is_path_leased(session.request.path)
            assert not session.released.is_set()
            assert sf.info(session.request.path).samplerate == 51200
        else:
            pump(ui_qapp, session.released.is_set)
        if outcome in ("ordinary", "streaming"):
            assert session.state == "completed"
            assert bool(previews) is (outcome == "streaming")
            host.run.assert_not_called()
            host.saved.assert_called_once()
            assert host.data_struct.wav_calibration_metadata == session.request.calibration_metadata
            raw, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
            assert rate == host.data_struct.sample_rate == 51200
            np.testing.assert_array_equal(raw, host.data_struct.store_wave_data_multi)
            assert np.all(raw[:, 0] == 8.25) and np.all(raw[:, 1] == 2.5)
        else:
            assert host.data_struct.store_wave_data_multi is None
            host.run.assert_not_called()
            host.saved.assert_not_called()
            host._send_recording_tcp_finish.assert_not_called()
            if outcome != "close":
                assert host.data_btn.isEnabled() and host.replayer_btn.isEnabled()
    finally:
        if session is not None and session.release_error is not None:
            # Restore the deliberately failed test lease, as the legacy service
            # tests do; the GUI under test never deletes or moves this path.
            session._cleanup_failed = False
            session.release_error = None
            service._release(session)
        bridge.shutdown()
        pump(ui_qapp, service.closed.is_set)
        assert all(not thread.is_alive() for thread in service.threads)


@pytest.mark.parametrize("streaming", [False, True])
def test_ve_records_with_corrupt_legacy_monitor_without_output(host_factory, streaming):
    host = host_factory()
    host.speaker = None
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(monitor_playback=True, monitor_gain_db="corrupt",
                  monitor_fade_in_ms={}, use_streaming_recording=streaming)
    host.judge_play_and_record()
    request = host._recording_process_session.request
    assert dict(request.monitor) == {}
    assert request.effective_streaming is streaming
    capture, audio = capture_audio(request)
    assert isinstance(capture.outcome, RecordingResult)
    assert (capture.snapshot(generation=1, sequence=1) is not None) is streaming
    assert len(audio.multi) == request.target_samples - request.trim_samples
