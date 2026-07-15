from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from base.data_struct.data_deal_struct import DataDealStruct
from ui.sequence import sequence_widget


class _Button:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


def _detail(start_freq=100):
    return {
        "sample_rate": 44100,
        "stimulus_info": {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "sample_rate": 44100,
            "start_freq": start_freq,
            "stop_freq": 1000,
            "total_time": 0.01,
            "repeat_times": 1,
            "amplitude": 0.5,
        },
    }


def _sequence(detail, mode="IMPORT_STIMULUS_AUDIO"):
    return [{"seq1": {"acq": {"mode": mode, "detail": detail}}}]


def _runtime_window(detail=None):
    detail = deepcopy(detail or _detail())
    mono = np.ones(320, dtype=np.float32)
    multi = mono.reshape(-1, 1)
    reference = np.ones(320, dtype=np.float32)
    data_struct = SimpleNamespace(
        sample_rate=32000,
        store_wave_data=mono,
        store_wave_data_multi=multi,
        audio_lenth=320,
        stimulus_data=reference,
        stimulus_info={"sample_rate": 32000, "total_time": 0.01},
        alignment_sample_count=320,
        wav_calibration_metadata={"source": "recording.wav"},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
        analysis_result_dict={"old": (True, "OK")},
    )
    window = SimpleNamespace(
        mode="IMPORT_STIMULUS_AUDIO",
        sequence_config=_sequence(detail),
        analysis_config={"display_sequence": []},
        data_struct=data_struct,
        recorded_path="recording.wav",
        recorded_signal_info={"file_path": "recording.wav"},
        data_btn=_Button(True),
        count_board=None,
        default_logger=SimpleNamespace(warning=lambda *_: None),
        update_using_file_combobox=lambda: None,
        init_fft_and_stft_flag=lambda: None,
        refresh_channel_windows=lambda: None,
        _clear_plot_area=lambda: None,
        _refresh_test_mode_availability=lambda: None,
        using_config_path="configs/sequence.json",
    )
    for name in (
        "_is_positive_runtime_integer",
        "_has_runtime_samples",
        "_has_imported_recording_runtime_state",
        "_has_import_stimulus_runtime_reference",
        "_refresh_import_stimulus_analysis_reference",
    ):
        setattr(window, name, MethodType(getattr(sequence_widget.SequenceWindow, name), window))
    return window


def test_analysis_only_config_update_preserves_import_runtime(monkeypatch):
    window = _runtime_window()
    original_recorded_path = window.recorded_path
    original_recorded_signal_info = window.recorded_signal_info
    original = {
        name: getattr(window.data_struct, name)
        for name in (
            "store_wave_data",
            "store_wave_data_multi",
            "stimulus_data",
            "stimulus_info",
            "wav_calibration_metadata",
        )
    }
    init_calls = []
    window.init_data_struct_stimulus_config = lambda: init_calls.append(True)

    def reload_config():
        window.analysis_config = {"display_sequence": ["SPL"]}

    window.get_sequence_config_from_json = reload_config
    sequence_widget.SequenceWindow.on_sequence_config_updated(window)

    assert init_calls == []
    assert window.data_struct.sample_rate == 32000
    assert window.data_struct.audio_lenth == 320
    for name, value in original.items():
        assert getattr(window.data_struct, name) is value
    assert window.recorded_path == original_recorded_path
    assert window.recorded_signal_info is original_recorded_signal_info
    assert window.data_btn.enabled is True


def test_stale_mode_state_preserves_same_import_stimulus_config_refresh():
    window = _runtime_window()
    window.mode = None
    original_recorded_path = window.recorded_path
    original_recorded_signal_info = window.recorded_signal_info
    original = {
        name: getattr(window.data_struct, name)
        for name in (
            "store_wave_data",
            "store_wave_data_multi",
            "stimulus_data",
            "stimulus_info",
            "wav_calibration_metadata",
        )
    }
    events = []

    def reload_config():
        window.sequence_config = _sequence(deepcopy(_detail()))
        window.analysis_config = {"display_sequence": ["SPL"]}
        window.mode = "IMPORT_STIMULUS_AUDIO"

    def clear_data():
        events.append("clear_data")
        window.data_struct.stimulus_data = None
        window.data_struct.stimulus_info = None

    window.get_sequence_config_from_json = reload_config
    window._clear_plot_area = lambda: events.append("clear_plot")
    window.data_struct.clear_data = clear_data
    window.refresh_channel_windows = lambda: events.append("refresh_channels")
    window.init_data_struct_stimulus_config = lambda: events.append("init_runtime")

    sequence_widget.SequenceWindow.on_sequence_config_updated(window)

    assert events == []
    assert window.data_struct.sample_rate == 32000
    assert window.data_struct.audio_lenth == 320
    for name, value in original.items():
        assert getattr(window.data_struct, name) is value
    assert window.recorded_path == original_recorded_path
    assert window.recorded_signal_info is original_recorded_signal_info
    assert window.data_btn.enabled is True


def test_stimulus_config_update_rebuilds_reference_at_imported_rate(monkeypatch):
    window = _runtime_window()
    original_mono = window.data_struct.store_wave_data
    original_multi = window.data_struct.store_wave_data_multi
    original_calibration = window.data_struct.wav_calibration_metadata
    original_recorded_path = window.recorded_path
    original_recorded_signal_info = window.recorded_signal_info
    calls = []

    def reload_config():
        window.sequence_config = _sequence(_detail(start_freq=200))
        window.mode = "IMPORT_STIMULUS_AUDIO"

    def build_reference(staged, detail, using_config_path=None, *, runtime_sample_rate, logger=None):
        calls.append((detail, using_config_path, runtime_sample_rate))
        staged.stimulus_data = np.full(320, 2.0, dtype=np.float32)
        staged.stimulus_info = {"sample_rate": runtime_sample_rate, "total_time": 0.01}
        staged.alignment_sample_count = 320
        return True

    window.get_sequence_config_from_json = reload_config
    window.init_data_struct_stimulus_config = lambda: pytest.fail("import runtime must not be reinitialized")
    monkeypatch.setattr(sequence_widget, "set_data_struct_analysis_reference_signal", build_reference)

    sequence_widget.SequenceWindow.on_sequence_config_updated(window)

    assert calls[0][2] == 32000
    assert window.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"]["sample_rate"] == 44100
    assert window.data_struct.store_wave_data is original_mono
    assert window.data_struct.store_wave_data_multi is original_multi
    assert window.data_struct.wav_calibration_metadata is original_calibration
    assert window.recorded_path == original_recorded_path
    assert window.recorded_signal_info is original_recorded_signal_info
    assert window.data_struct.stimulus_info == {"sample_rate": 32000, "total_time": 0.01}
    assert window.data_struct.alignment_sample_count == 320
    assert window.data_btn.enabled is True


@pytest.mark.parametrize("failure", ["false", "raise"])
def test_failed_stimulus_refresh_clears_only_reference(monkeypatch, failure):
    window = _runtime_window()
    original_mono = window.data_struct.store_wave_data
    original_multi = window.data_struct.store_wave_data_multi
    original_calibration = window.data_struct.wav_calibration_metadata
    warnings = []

    def reload_config():
        window.sequence_config = _sequence(_detail(start_freq=200))

    def fail_reference(*args, **kwargs):
        if failure == "raise":
            raise RuntimeError("reference failed")
        return False

    window.get_sequence_config_from_json = reload_config
    window.init_data_struct_stimulus_config = lambda: pytest.fail("full import state must not be cleared")
    monkeypatch.setattr(sequence_widget, "set_data_struct_analysis_reference_signal", fail_reference)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.on_sequence_config_updated(window)

    assert len(warnings) == 1
    assert warnings[0][1] == "提示"
    expected_message = (
        "加载分析参考激励失败: reference failed"
        if failure == "raise"
        else "加载分析参考激励失败，请检查激励配置。"
    )
    assert warnings[0][2] == expected_message
    assert window.data_struct.store_wave_data is original_mono
    assert window.data_struct.store_wave_data_multi is original_multi
    assert window.data_struct.wav_calibration_metadata is original_calibration
    assert window.data_struct.sample_rate == 32000
    assert window.data_struct.audio_lenth == 320
    assert window.data_struct.stimulus_data is None
    assert window.data_struct.stimulus_info is None
    assert not hasattr(window.data_struct, "alignment_sample_count")
    assert window.recorded_path == "recording.wav"
    assert window.data_btn.enabled is False


def test_mode_change_uses_existing_full_reset_path_and_clears_import_runtime():
    window = _runtime_window()
    events = []

    def reload_config():
        window.sequence_config = _sequence({}, mode="IMPORT_AUDIO")
        window.mode = "IMPORT_AUDIO"

    def clear_data():
        events.append("clear_data")
        DataDealStruct.clear_data(window.data_struct)

    def init_runtime():
        events.append("init_runtime")
        sequence_widget.SequenceWindow.init_data_struct_stimulus_config(window)

    window.get_sequence_config_from_json = reload_config
    window._clear_plot_area = lambda: events.append("clear_plot")
    window.data_struct.clear_data = clear_data
    window.refresh_channel_windows = lambda: events.append("refresh_channels")
    window.init_data_struct_stimulus_config = init_runtime
    window._refresh_import_stimulus_analysis_reference = lambda *_: pytest.fail(
        "mode changes must not refresh the old import reference"
    )

    sequence_widget.SequenceWindow.on_sequence_config_updated(window)

    assert events == ["clear_plot", "clear_data", "refresh_channels", "init_runtime"]
    assert window.data_btn.enabled is False
    assert window.data_struct.sample_rate is None
    assert window.data_struct.audio_lenth is None
    assert window.data_struct.store_wave_data is None
    assert window.data_struct.store_wave_data_multi is None
    assert window.data_struct.stimulus_data is None
    assert window.data_struct.stimulus_info is None
    assert not hasattr(window.data_struct, "alignment_sample_count")
    assert window.recorded_path is None
    assert window.recorded_signal_info is None
    assert window.data_struct.wav_calibration_metadata is None
    assert window.data_struct.wav_calibration_metadata_authoritative is False
    assert window.data_struct.wav_calibration_warning_shown is False
    assert window._has_imported_recording_runtime_state() is False
    assert window._has_import_stimulus_runtime_reference() is False


@pytest.mark.parametrize("mode", ["IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"])
def test_import_mode_stimulus_config_init_preserves_imported_recording_sample_rate(mode):
    window = _runtime_window()
    window.sequence_config = _sequence({}, mode=mode)
    original_mono = window.data_struct.store_wave_data
    original_multi = window.data_struct.store_wave_data_multi
    original_calibration = window.data_struct.wav_calibration_metadata

    sequence_widget.SequenceWindow.init_data_struct_stimulus_config(window)

    assert window.data_struct.sample_rate == 32000
    assert window.data_struct.audio_lenth == 320
    assert window.data_struct.store_wave_data is original_mono
    assert window.data_struct.store_wave_data_multi is original_multi
    assert window.data_struct.wav_calibration_metadata is original_calibration
    assert window.data_struct.wav_calibration_metadata_authoritative is True
    assert window.data_struct.wav_calibration_warning_shown is True
    assert window.data_struct.stimulus_data is None
    assert window.data_struct.stimulus_info is None


READINESS_WARNING = "分析参考激励尚未就绪或采样率与导入音频不一致，请检查激励配置后重试。"


def _run_window():
    window = _runtime_window()
    window.analysis_window = [SimpleNamespace(name="previous-analysis-window")]
    window._analysis_result_summary_window = SimpleNamespace(name="previous-summary")
    window.analysis_config = {
        "display_sequence": ["item"],
        "item": {"type": "FAKE"},
    }
    window.instantiated_analysis = []

    def instance_analysis_class(key, item_type, config):
        assert window.data_struct.analysis_result_dict == {}
        window.instantiated_analysis.append((key, item_type, config))

    window.instance_analysis_class = instance_analysis_class
    window._handle_post_analysis_exports = lambda: None
    window.count_board = SimpleNamespace(mode="analysis")
    window.screen = lambda: SimpleNamespace(
        size=lambda: SimpleNamespace(width=lambda: 1200, height=lambda: 800)
    )
    window._maybe_show_analysis_result_summary = lambda *args: None
    window._send_tcp_analysis_result_callback = lambda: None
    if hasattr(sequence_widget.SequenceWindow, "_validate_import_stimulus_analysis_readiness"):
        window._validate_import_stimulus_analysis_readiness = MethodType(
            sequence_widget.SequenceWindow._validate_import_stimulus_analysis_readiness,
            window,
        )
    return window


def _assert_previous_output_state_preserved(window, previous_windows, previous_summary):
    assert window.data_struct.analysis_result_dict == {"old": (True, "OK")}
    assert window.analysis_window is previous_windows
    assert window._analysis_result_summary_window is previous_summary
    assert window.instantiated_analysis == []


def test_run_with_missing_reference_warns_without_clearing_previous_results(monkeypatch):
    window = _run_window()
    previous_windows = window.analysis_window
    previous_summary = window._analysis_result_summary_window
    window.data_struct.stimulus_info = None
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.run(window)

    assert warnings == [(window, "提示", READINESS_WARNING)]
    _assert_previous_output_state_preserved(window, previous_windows, previous_summary)


def test_run_with_reference_rate_mismatch_warns_without_analysis(monkeypatch):
    window = _run_window()
    previous_windows = window.analysis_window
    previous_summary = window._analysis_result_summary_window
    window.data_struct.stimulus_info = {"sample_rate": 44100, "total_time": 0.01}
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.run(window)

    assert warnings == [(window, "提示", READINESS_WARNING)]
    _assert_previous_output_state_preserved(window, previous_windows, previous_summary)


@pytest.mark.parametrize("total_time", [None, 0, float("nan")])
def test_run_with_invalid_total_time_warns_without_clearing_previous_results(monkeypatch, total_time):
    window = _run_window()
    previous_windows = window.analysis_window
    previous_summary = window._analysis_result_summary_window
    window.data_struct.stimulus_info = {"sample_rate": 32000, "total_time": total_time}
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.run(window)

    assert warnings == [(window, "提示", READINESS_WARNING)]
    _assert_previous_output_state_preserved(window, previous_windows, previous_summary)


def test_run_retains_length_mismatch_gate_before_clearing_results(monkeypatch):
    window = _run_window()
    previous_windows = window.analysis_window
    previous_summary = window._analysis_result_summary_window
    window.data_struct.audio_lenth = 319
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.run(window)

    assert warnings == [
        (
            window,
            "音频长度校验失败",
            "导入音频长度(319)\n与激励信号长度(320)不一致！无法分析！",
        )
    ]
    _assert_previous_output_state_preserved(window, previous_windows, previous_summary)


def test_run_clears_previous_results_after_all_import_gates_pass(monkeypatch):
    window = _run_window()
    warnings = []
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", lambda *args: warnings.append(args))

    sequence_widget.SequenceWindow.run(window)

    assert warnings == []
    assert window.data_struct.analysis_result_dict == {}
    assert window.analysis_window == []
    assert window._analysis_result_summary_window is None
    assert window.instantiated_analysis == [
        ("item", "FAKE", window.analysis_config["item"])
    ]
