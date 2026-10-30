"""Exercise real output preparation and start boundaries without opening audio devices."""
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QLineEdit

from base.data_struct.data_deal_struct import DataDealStruct
from base.load_config import LoadUiConfig
from base.output_calibration import OutputCalibrationCache
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from ui import operation_sequence
from ui.sequence import sequence_widget
from unit_test.base.test_hardware_management import runtime_device
from unit_test.ui.test_speaker_calibration_db_consumers import _registered_speaker_db, _select_speaker
from unit_test.ui.test_sequence_widget_sample_rate_authority import _window
from unit_test.ui.test_sequence_widget_sn_lock import FakeButton, FakeLineEdit


@pytest.fixture(scope="session", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.25, 0.0], max_voltage=2.0)
    speaker = dict(speaker, samplerate=48000, index=7, hostapi=1)
    reads, predictions, warnings = [], [], []
    read = SoundcardCalibrationManager.load_output_amplitude_calibration
    predict = SoundcardCalibrationManager.predict_amplitude

    def counted_read(manager):
        reads.append(manager.speaker_hardware_id)
        return read(manager)

    def counted_predict(coefficients, voltage):
        predictions.append(voltage)
        return predict(coefficients, voltage)

    monkeypatch.setattr(SoundcardCalibrationManager, "load_output_amplitude_calibration", counted_read)
    monkeypatch.setattr(SoundcardCalibrationManager, "predict_amplitude", staticmethod(counted_predict))
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(sequence_widget, "get_recorded_info", lambda *args: ("unused.wav", {"labels": args[-1]}))
    return SimpleNamespace(repo=repo, speaker=speaker, reads=reads, predictions=predictions, warnings=warnings)


def window(calibrated, *, amplitude=None, voltage=1.0, mode="PLAY_AND_RECORD"):
    info = {"voltage": voltage, "voltage_type": "RMS", "total_time": 0.01, "sample_rate": 48000}
    if amplitude is not None:
        info["amplitude"] = amplitude
    win = _window(mode, {"stimulus_info": info, "recording_start_delay_ms": 125.0}, speaker=calibrated.speaker)
    win.data_struct.stimulus_info = deepcopy(info)
    win.data_struct.stimulus_data = np.array([0.2, -0.4], dtype=np.float32)
    win.data_struct.clear_data = MethodType(DataDealStruct.clear_data, win.data_struct)
    win._output_calibration_cache = OutputCalibrationCache()
    win._output_calibration_config_revision = 0
    for name in ("reset_work_pram", "_prepare_output_calibration", "invalidate_output_calibration", "get_sequence_config_from_json"):
        setattr(win, name, MethodType(getattr(sequence_widget.SequenceWindow, name), win))
    return win


@pytest.mark.parametrize("amplitude", [None, 0.9])
def test_preparation_uses_selected_speaker_and_preserves_raw_waveform(calibrated, monkeypatch, amplitude):
    other = calibrated.repo.register_asset(runtime_device(name="Global", inputs=0, outputs=1, index=8), "Windows WASAPI", "Global", 48000)
    calibrated.repo.update_output_amplitude_calibration(other["hardware_id"], [0.8, 0.0], max_voltage=2.0)
    _select_speaker(monkeypatch, other["hardware_id"])
    win = window(calibrated, amplitude=amplitude)
    before = deepcopy(win.sequence_config)
    raw = win.data_struct.stimulus_data.copy()

    stimulus, recorded, rate = win.reset_work_pram("first")

    assert stimulus["amplitude"] == 0.25
    np.testing.assert_array_equal(stimulus["data"], raw)
    assert recorded["recording_start_delay_frames"] == 6000
    assert recorded["input_channels"] == [0]
    assert rate == 48000
    assert calibrated.reads == [win.speaker["hardware_id"]]
    assert calibrated.predictions == [1.0]
    assert win.sequence_config == before


def _start_harness(win, *, streaming):
    win.tcp_flag = False
    win._record_workflow_busy = False
    win.player_status_flag = False
    win.analysis_window = []
    win._analysis_result_summary_window = None
    win.last_play_count = 1
    win.current_recorded_count = 1
    win._pending_recorded_count = None
    win.lineedit_s_or_n = FakeLineEdit("SN001")
    win.player_btn, win.replayer_btn, win.data_btn = FakeButton(), FakeButton(), FakeButton()
    win.default_logger = SimpleNamespace(error=lambda *a: pytest.fail(str(a)))
    win.checked_work_status_message = lambda: False
    win._validate_sn_regex_before_start = lambda: True
    win._set_sn_input_recording_read_only = win.lineedit_s_or_n.setReadOnly
    win._clear_plot_area = lambda: None
    win._cleanup_streaming_resources = lambda: None
    win.update_player_btn_is_playing = lambda: win.player_btn.setEnabled(False)
    win.update_player_btn_is_paused = lambda: win.player_btn.setEnabled(True)
    win._should_use_streaming_recording = lambda: streaming
    for name in ("judge_play_and_record", "_ensure_pending_recorded_count", "_clear_pending_recorded_count"):
        setattr(win, name, MethodType(getattr(sequence_widget.SequenceWindow, name), win))
    starts, attempts = [], []
    win._begin_recording_output_attempt = lambda replay: attempts.append(replay)

    def capture(kind, stimulus, recorded, rate):
        starts.append((kind, stimulus, recorded, rate))
        # Audio completion is outside this test's boundary; release the workflow for the next start.
        win._record_workflow_busy = win.player_status_flag = False
        win._set_sn_input_recording_read_only(False)

    win._start_streaming_recording = lambda *args: capture("stream", *args)
    win._start_blocking_recording = lambda *args: capture("blocking", *args)
    return starts, attempts


@pytest.mark.parametrize("streaming,replay", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("amplitude", [None, 0.9])
def test_unchanged_runs_cache_across_start_paths_and_routine_cleanup(calibrated, streaming, replay, amplitude):
    win = window(calibrated, amplitude=amplitude)
    starts, _ = _start_harness(win, streaming=streaming)
    win.judge_play_and_record("first", is_replay=replay)
    win.data_struct.stimulus_info["amplitude"] = 0.123
    win.lineedit_count = SimpleNamespace(text=lambda: "99")
    win.judge_play_and_record("changed-label", is_replay=replay)
    assert [s[1]["amplitude"] for s in starts] == [0.25, 0.25]
    assert len(calibrated.reads) == len(calibrated.predictions) == 1


@pytest.mark.parametrize("streaming,replay", [(False, False), (True, False), (False, True), (True, True)])
def test_range_rejection_restores_workflow_and_correction_can_retry(calibrated, streaming, replay):
    win = window(calibrated, amplitude=0.9, voltage=3.0)
    starts, attempts = _start_harness(win, streaming=streaming)
    win.judge_play_and_record(is_replay=replay)
    assert starts == attempts == []
    assert win.current_recorded_count == 1
    assert win._pending_recorded_count is None
    assert not win._record_workflow_busy and not win.player_status_flag
    assert not win.lineedit_s_or_n.isReadOnly()
    assert all(btn.isEnabled() for btn in (win.player_btn, win.replayer_btn, win.data_btn))
    assert "3 V" in calibrated.warnings[-1] and "2 V" in calibrated.warnings[-1]
    assert calibrated.predictions == []
    win.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"]["voltage"] = 2.0
    win.judge_play_and_record(is_replay=replay)
    assert starts[0][1]["amplitude"] == 0.5


@pytest.mark.parametrize("change", ["voltage", "voltage_type", "config", "path", "invalidate"])
def test_context_changes_rebuild_once(calibrated, change):
    win = window(calibrated, amplitude=0.9)
    win.reset_work_pram("first")
    info = win.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"]
    if change == "voltage":
        info["voltage"] = 2.0
    elif change == "voltage_type":
        info["voltage_type"] = "Peak"
    elif change == "config":
        win.sequence_config = deepcopy(win.sequence_config)
    elif change == "path":
        win.using_config_path = "other.json"
    else:
        win.invalidate_output_calibration()
    win.reset_work_pram("next")
    win.reset_work_pram("repeat")
    assert len(calibrated.reads) == len(calibrated.predictions) == 2


def test_reload_and_clear_invalidate_even_same_config_object(calibrated, monkeypatch):
    win = window(calibrated, amplitude=0.9)
    win.count_board = None
    win._set_sequence_config_available_state = lambda value: None
    config = win.sequence_config
    monkeypatch.setattr(LoadUiConfig, "load_sequence_config_from_json", lambda *a: (0, config))
    win.reset_work_pram("first")
    win.get_sequence_config_from_json()
    win.reset_work_pram("reload")
    assert len(calibrated.reads) == 2
    monkeypatch.setattr(LoadUiConfig, "load_sequence_config_from_json", lambda *a: (1, None))
    win.get_sequence_config_from_json()
    assert win.sequence_config == []
    win.sequence_config = config
    win.reset_work_pram("restored")
    assert len(calibrated.reads) == 3


def test_failed_rebuild_discards_old_success_and_retries_missing_calibration(calibrated):
    win = window(calibrated, amplitude=0.9)
    assert win.reset_work_pram("first")[0]["amplitude"] == 0.25
    calibrated.repo.update_output_amplitude_calibration(win.speaker["hardware_id"], [0.1, 0.0], max_voltage=0.5)
    win.invalidate_output_calibration()
    assert win.reset_work_pram("invalid") == (None, None, None)
    assert win.reset_work_pram("invalid-again") == (None, None, None)
    calibrated.repo.update_output_amplitude_calibration(win.speaker["hardware_id"], [0.1, 0.0], max_voltage=2.0)
    assert win.reset_work_pram("recovered")[0]["amplitude"] == 0.1
    assert len(calibrated.reads) == 4
    win.speaker = dict(win.speaker, hardware_id="unregistered")
    assert win.reset_work_pram("missing")[0]["amplitude"] == 0.0
    assert win.reset_work_pram("missing-again")[0]["amplitude"] == 0.0
    assert len(calibrated.reads) == 6


def test_new_instance_has_empty_cache_and_record_only_skips_preparation(calibrated):
    for _ in range(2):
        window(calibrated, amplitude=0.9).reset_work_pram("run")
    assert len(calibrated.reads) == 2
    for mode in ("RECORD_ONLY", "IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"):
        win = window(calibrated, mode=mode)
        win.data_struct.sample_rate = 48000
        win.data_struct.stimulus_data = None
        win.reset_work_pram("nonoutput")
    assert len(calibrated.reads) == 2


def test_same_hardware_metadata_mic_and_output_path_changes_keep_cache(calibrated):
    win = window(calibrated)
    starts, _ = _start_harness(win, streaming=False)
    win.judge_play_and_record()
    win.speaker = dict(win.speaker, index=20, name="Renamed")
    win.mic = dict(win.mic, index=21)
    win._should_use_streaming_recording = lambda: True
    win.judge_play_and_record()
    win.judge_play_and_record(is_replay=True)
    assert [s[0] for s in starts] == ["blocking", "stream", "stream"]
    assert [s[1]["amplitude"] for s in starts] == [0.25] * 3
    assert len(calibrated.reads) == len(calibrated.predictions) == 1


def test_switch_to_lower_limit_speaker_cannot_reuse_old_success(calibrated):
    win = window(calibrated)
    win.reset_work_pram("first")
    other = calibrated.repo.register_asset(runtime_device(name="Low limit", inputs=0, outputs=1, index=8), "Windows WASAPI", "Low limit", 48000)
    calibrated.repo.update_output_amplitude_calibration(other["hardware_id"], [0.8, 0.0], max_voltage=0.5)
    win.speaker = dict(win.speaker, hardware_id=other["hardware_id"])
    assert win.reset_work_pram("switched") == (None, None, None)
    assert calibrated.predictions == [1.0]
    win.speaker = calibrated.speaker
    assert win.reset_work_pram("returned")[0]["amplitude"] == 0.25
    assert len(calibrated.reads) == 3


@pytest.mark.parametrize("event", ["edit", "switch"])
def test_applied_config_callbacks_reload_and_invalidate(calibrated, monkeypatch, event):
    win = window(calibrated)
    win.reset_work_pram("first")
    config = win.sequence_config
    win.count_board = None
    win._set_sequence_config_available_state = lambda value: None
    win.update_using_file_combobox = lambda: None
    win.init_data_struct_stimulus_config = lambda: None
    win.init_fft_and_stft_flag = lambda: None
    win.refresh_channel_windows = lambda: None
    win.update_player_btn_is_paused = lambda: None
    win.player_status_flag = False
    win.replayer_btn, win.data_btn = FakeButton(), FakeButton()
    win.lineedit_s_or_n = QLineEdit("SN001")
    win.using_file_combobox = SimpleNamespace(currentData=lambda: "new.json", clearFocus=lambda: None)
    win.default_logger = SimpleNamespace(warning=lambda *args: pytest.fail(str(args)))
    monkeypatch.setattr(LoadUiConfig, "load_sequence_config_from_json", lambda *a: (0, config))
    monkeypatch.setattr(LoadUiConfig, "update_using_config_path", lambda *a: None)
    if event == "edit":
        sequence_widget.SequenceWindow.on_sequence_config_updated(win)
    else:
        sequence_widget.SequenceWindow.on_using_file_combobox_changed(win, "new")
    win.reset_work_pram("after")
    win.reset_work_pram("repeat")
    assert len(calibrated.reads) == 2


@pytest.mark.parametrize("voltage", [1.0, 3.0])
def test_golden_sample_prepares_own_amplitude_and_preserves_ordinary_state(calibrated, monkeypatch, voltage):
    win = window(calibrated, voltage=voltage)
    ds = win.data_struct
    ds.sample_rate = 44100
    old_info, old_data = ds.stimulus_info, ds.stimulus_data
    ds.store_wave_data = np.array([0.5])
    old_recording = ds.store_wave_data
    seq = SimpleNamespace(detail=win.sequence_config[0]["seq1"]["acq"]["detail"], analysis_list={"display_sequence": ["FR"], "FR": {"type": "FR", "golden_sample_checked": True}})
    selector = SimpleNamespace(select_list=SimpleNamespace(config=[seq], data_struct=ds), mic=win.mic, speaker=win.speaker, using_config_path="seq.json", default_logger=SimpleNamespace(error=lambda *a: None))

    def generate(data_struct, detail, **kwargs):
        data_struct.stimulus_info = deepcopy(detail["stimulus_info"])
        data_struct.stimulus_data = np.array([0.2, -0.4], dtype=np.float32)
        return True

    selector.set_data_struct_stimulus_signal = generate
    captures = []
    monkeypatch.setattr(operation_sequence.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(operation_sequence, "build_recording_wav_calibration_metadata", lambda *a, **k: {})
    monkeypatch.setattr(operation_sequence.SoundcardAudioProcessor, "sd_play_rec", lambda self, recorded, stimulus, *a, **k: captures.append(stimulus) or (1, None))
    for _ in range(2):
        operation_sequence.AnalysisModelSelect.record_golden_sample_btn_clicked(selector)
        assert ds.sample_rate == 44100
        assert ds.stimulus_info is old_info and ds.stimulus_data is old_data
        assert ds.store_wave_data is old_recording
    assert len(calibrated.reads) == 2
    if voltage > 2.0:
        assert captures == []
        assert "3 V" in calibrated.warnings[-1]
    else:
        assert [s["amplitude"] for s in captures] == [0.25, 0.25]
        np.testing.assert_array_equal(captures[0]["data"], old_data)


def test_builder_missing_amplitude_nonoutput_fallback_is_zero():
    ds = SimpleNamespace(stimulus_data=np.array([0.2]), stimulus_info={}, sample_rate=48000)
    stimulus, _ = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(ds)
    assert stimulus["amplitude"] == 0.0
