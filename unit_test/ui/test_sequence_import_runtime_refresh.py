from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from base.data_struct.data_deal_struct import DataDealStruct
from base.soundcard_calibration_manager import AnalysisV2paBatch
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


def _run_window(mode="IMPORT_STIMULUS_AUDIO"):
    window = _runtime_window()
    window.mode = mode
    window.sequence_config = _sequence({}, mode=mode)
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


def test_run_skips_failed_spec_and_continues_later_analysis_and_exports():
    window = _run_window()
    events = []

    class FailedSpec:
        _sequence_analysis_key = "spec"

        def calculate_spec(self):
            events.append("spec_calculate")
            return False

        def show(self):
            events.append("spec_show")

    class SuccessfulFft:
        _sequence_analysis_key = "fft"

        def calculate_fft(self):
            events.append("fft_calculate")
            return True

        def show(self):
            events.append("fft_show")

        def setGeometry(self, *args):
            pass

        def setMinimumSize(self, *args):
            pass

        def installEventFilter(self, *args):
            pass

    window.analysis_config = {
        "display_sequence": ["spec", "fft"],
        "spec": {"type": "Spec"},
        "fft": {"type": "FFT"},
    }

    def instance_analysis_class(key, item_type, config):
        if key == "spec":
            window.analysis_window.append(FailedSpec())
        else:
            window.analysis_window.append(SuccessfulFft())

    window.instance_analysis_class = instance_analysis_class
    window._analysis_window_key_by_obj = {}
    window._get_analysis_window_geometry = lambda _key: None
    window._set_analysis_window_geometry = lambda *_args: None
    window._handle_post_analysis_exports = lambda: events.append("export")

    sequence_widget.SequenceWindow.run(window)

    assert events == ["spec_calculate", "fft_calculate", "fft_show", "export"]


def _configure_live_batch_run(window, events, raw_channels=(0, 1)):
    class SuccessfulSpl:
        def __init__(self, key):
            self._sequence_analysis_key = key

        def calculate_spl(self):
            events.append(f"{self._sequence_analysis_key}:calculate")
            return True

        def show(self):
            events.append(f"{self._sequence_analysis_key}:show")

        def setGeometry(self, *args):
            pass

        def setMinimumSize(self, *args):
            pass

        def installEventFilter(self, *args):
            pass

    raw_channels = tuple(raw_channels)
    item_keys = (
        ["first", "second"]
        if raw_channels == (0, 1)
        else [f"item-{index}" for index in range(len(raw_channels))]
    )
    window.analysis_config = {
        "display_sequence": item_keys,
        **{
            key: {"type": "SPL", "analysis_channel": raw_channel}
            for key, raw_channel in zip(item_keys, raw_channels)
        },
    }

    def instance_analysis_class(key, item_type, config):
        batch = window._analysis_v2pa_batch
        preparation = batch.resolve(config["analysis_channel"])
        assert preparation.factor is not None
        window.analysis_window.append(SuccessfulSpl(key))

    window.instance_analysis_class = instance_analysis_class
    window._analysis_window_key_by_obj = {}
    window._get_analysis_window_geometry = lambda _key: None
    window._set_analysis_window_geometry = lambda *_args: None
    window._handle_post_analysis_exports = lambda: events.append("export")


def _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls, events=None):
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: False,
    )

    def show(parent, text, logger=None):
        warning_calls.append((parent, text))
        if events is not None:
            events.append("warning")

    monkeypatch.setattr(
        sequence_widget,
        "_show_uncalibrated_microphone_warning",
        show,
    )


@pytest.mark.parametrize(
    "checked_after_exec",
    [False, True],
)
def test_uncalibrated_microphone_warning_uses_dedicated_checkbox_dialog(
    monkeypatch,
    checked_after_exec,
):
    events = []
    created_message_boxes = []
    created_checkboxes = []
    parent = object()
    logger = object()

    class FakeCheckBox:
        def __init__(self, text):
            self.text = text
            self.checked = False
            created_checkboxes.append(self)

        def isChecked(self):
            return self.checked

    class FakeMessageBox:
        Warning = object()

        def __init__(self, supplied_parent):
            self.parent = supplied_parent
            self.icon = None
            self.title = None
            self.text = None
            self.checkbox = None
            self.standard_buttons_were_set = False
            created_message_boxes.append(self)

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, text):
            self.text = text

        def setCheckBox(self, checkbox):
            self.checkbox = checkbox

        def setStandardButtons(self, _buttons):
            self.standard_buttons_were_set = True

        def exec_(self):
            assert self.checkbox.checked is False
            events.append("exec")
            self.checkbox.checked = checked_after_exec

    def save(*, logger=None):
        assert logger is globals_logger
        events.append("save")
        return False

    globals_logger = logger
    monkeypatch.setattr(sequence_widget, "MessageBox", FakeMessageBox)
    monkeypatch.setattr(sequence_widget, "CheckBox", FakeCheckBox, raising=False)
    monkeypatch.setattr(
        sequence_widget,
        "save_uncalibrated_microphone_warning_suppressed",
        save,
        raising=False,
    )

    sequence_widget._show_uncalibrated_microphone_warning(
        parent,
        "麦克风未进行校准，结果仅供参考。",
        logger=logger,
    )

    assert len(created_message_boxes) == 1
    message_box = created_message_boxes[0]
    assert message_box.parent is parent
    assert message_box.icon is FakeMessageBox.Warning
    assert message_box.title == "提示"
    assert message_box.text == "麦克风未进行校准，结果仅供参考。"
    assert message_box.checkbox is created_checkboxes[0]
    assert message_box.standard_buttons_were_set is False
    assert created_checkboxes[0].text == "不在提示"
    assert events == (["exec", "save"] if checked_after_exec else ["exec"])


def test_suppressed_target_only_skips_warning_and_continues_analysis(monkeypatch):
    events = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events, raw_channels=(0,))

    def resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "AnalysisV2paBatch",
        lambda: AnalysisV2paBatch(resolver=resolver),
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: True,
        raising=False,
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: events.append("ordinary-warning"),
    )

    sequence_widget.SequenceWindow.run(window)

    assert events == ["item-0:calculate", "item-0:show", "export"]


def test_suppressed_target_preserves_independent_diagnostic(monkeypatch):
    events = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events, raw_channels=(0,))

    def resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        warn_callback("校准数据诊断")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "AnalysisV2paBatch",
        lambda: AnalysisV2paBatch(resolver=resolver),
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: True,
        raising=False,
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "提示", "校准数据诊断")]


def test_record_only_suppression_omits_channel_context_but_preserves_diagnostic(
    monkeypatch,
):
    events = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(2,))

    def resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        warn_callback("校准数据诊断")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )
    monkeypatch.setattr(
        sequence_widget,
        "AnalysisV2paBatch",
        lambda resolver=None: AnalysisV2paBatch(resolver=resolver),
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: True,
        raising=False,
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "提示", "校准数据诊断")]


def _configure_production_pd_run(window, monkeypatch, events):
    created = []

    class FakePd:
        _supports_pre_resolved_v2pa_factor = True

        def __init__(self, key):
            self.key = key
            created.append(self)
            events.append("pd:construct")

        def calculate_peak_detection(self):
            events.append("pd:calculate")

        def show(self):
            events.append("pd:show")

        def setGeometry(self, *args):
            pass

        def setMinimumSize(self, *args):
            pass

        def installEventFilter(self, *args):
            pass

    window.analysis_config = {
        "display_sequence": ["pd"],
        "pd": {"type": "PD", "analysis_channel": 0},
    }
    window.analysis_types_requiring_v2pa = {"PD", "ED"}
    window.instance_analysis_class = MethodType(
        sequence_widget.SequenceWindow.instance_analysis_class,
        window,
    )
    window._analysis_window_key_by_obj = {}
    window._get_analysis_window_geometry = lambda _key: None
    window._set_analysis_window_geometry = lambda *_args: None
    window._handle_post_analysis_exports = lambda: events.append("export")
    monkeypatch.setattr(sequence_widget, "get_class_mapping", lambda: {"PD": FakePd})
    return created


def test_non_batched_pd_unsuppressed_target_uses_dedicated_warning(monkeypatch):
    events = []
    warning_calls = []
    ordinary_warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    created = _configure_production_pd_run(window, monkeypatch, events)

    def resolver(raw_channel, warn_callback=None):
        events.append("pd:resolve")
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls, events)
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: ordinary_warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "麦克风未进行校准，结果仅供参考。")]
    assert ordinary_warning_calls == []
    assert created[0].v2pa_factor == 1.0
    assert created[0]._v2pa_raw_analysis_channel == 0
    assert created[0]._use_pre_resolved_v2pa_factor is True
    assert events == [
        "pd:construct",
        "pd:resolve",
        "warning",
        "pd:calculate",
        "pd:show",
        "export",
    ]


def test_non_batched_pd_suppressed_target_shows_no_dialog(monkeypatch):
    events = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    created = _configure_production_pd_run(window, monkeypatch, events)

    def resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: True,
    )
    monkeypatch.setattr(
        sequence_widget,
        "_show_uncalibrated_microphone_warning",
        lambda *args, **kwargs: pytest.fail("suppressed target must not use dedicated dialog"),
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == []
    assert created[0].v2pa_factor == 1.0
    assert events == ["pd:construct", "pd:calculate", "pd:show", "export"]


def test_non_batched_ed_uses_nested_channel_and_marks_prepared_factor(monkeypatch):
    events = []
    resolve_calls = []
    created = []
    window = _run_window(mode="PLAY_AND_RECORD")

    class FakeEd:
        _supports_pre_resolved_v2pa_factor = True

        def __init__(self, key):
            self._sequence_analysis_key = key
            created.append(self)

        def calculate_pipeline_pd_pm(self):
            events.append("ed:calculate")

        def show(self):
            events.append("ed:show")

        def setGeometry(self, *args):
            pass

        def setMinimumSize(self, *args):
            pass

        def installEventFilter(self, *args):
            pass

    window.analysis_config = {
        "display_sequence": ["ed"],
        "ed": {
            "type": "ED",
            "head": {"config": {"analysis_channel": 7}},
            "tail": {"config": {}},
        },
    }
    window.analysis_types_requiring_v2pa = {"PD", "ED"}
    window.instance_analysis_class = MethodType(
        sequence_widget.SequenceWindow.instance_analysis_class,
        window,
    )
    window._analysis_window_key_by_obj = {}
    window._get_analysis_window_geometry = lambda _key: None
    window._set_analysis_window_geometry = lambda *_args: None
    window._handle_post_analysis_exports = lambda: events.append("export")
    monkeypatch.setattr(sequence_widget, "get_class_mapping", lambda: {"ED": FakeEd})

    def resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 12.5

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )

    sequence_widget.SequenceWindow.run(window)

    assert resolve_calls == [7]
    assert created[0].v2pa_factor == 12.5
    assert created[0]._v2pa_raw_analysis_channel == 7
    assert created[0]._use_pre_resolved_v2pa_factor is True
    assert events == ["ed:calculate", "ed:show", "export"]


def test_non_batched_pd_suppression_preserves_independent_diagnostic(monkeypatch):
    events = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_production_pd_run(window, monkeypatch, events)

    def resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        warn_callback("校准数据诊断")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: True,
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: (warning_calls.append(args), events.append("ordinary-warning")),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "提示", "校准数据诊断")]
    assert events == [
        "pd:construct",
        "ordinary-warning",
        "pd:calculate",
        "pd:show",
        "export",
    ]


def test_non_batched_value_error_is_deferred_until_all_items_are_prepared(monkeypatch):
    events = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")

    class FakePd:
        def __init__(self, key):
            events.append("pd:construct")

    class LaterAnalysis:
        def __init__(self, key):
            self._sequence_analysis_key = key
            events.append("later:construct")

        def calculate_spl(self):
            events.append("later:calculate")
            return True

        def show(self):
            events.append("later:show")

        def setGeometry(self, *args):
            pass

        def setMinimumSize(self, *args):
            pass

        def installEventFilter(self, *args):
            pass

    window.analysis_config = {
        "display_sequence": ["pd", "later"],
        "pd": {"type": "PD", "analysis_channel": 0},
        "later": {"type": "FAKE"},
    }
    window.analysis_types_requiring_v2pa = {"PD", "ED"}
    window.instance_analysis_class = MethodType(
        sequence_widget.SequenceWindow.instance_analysis_class,
        window,
    )
    window._analysis_window_key_by_obj = {}
    window._get_analysis_window_geometry = lambda _key: None
    window._set_analysis_window_geometry = lambda *_args: None
    window._handle_post_analysis_exports = lambda: events.append("export")
    monkeypatch.setattr(
        sequence_widget,
        "get_class_mapping",
        lambda: {"PD": FakePd, "FAKE": LaterAnalysis},
    )

    def resolver(raw_channel, warn_callback=None):
        events.append("pd:resolve")
        raise ValueError("校准系数格式无效")

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        resolver,
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: (warning_calls.append(args), events.append("ordinary-warning")),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "提示", "校准系数格式无效")]
    assert events == [
        "pd:construct",
        "pd:resolve",
        "later:construct",
        "ordinary-warning",
        "later:calculate",
        "later:show",
        "export",
    ]


def test_record_only_run_lists_uncalibrated_channels_in_first_seen_order_before_calculation(
    monkeypatch,
):
    events = []
    resolve_calls = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(2, 0, 2))

    def fake_resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls, events)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [
        (
            window,
            "麦克风未进行校准，结果仅供参考。\n"
            "未校准通道：\n"
            "• In3\n"
            "• In1",
        )
    ]
    assert resolve_calls == [2, 0]
    assert events == [
        "warning",
        "item-0:calculate",
        "item-0:show",
        "item-1:calculate",
        "item-1:show",
        "item-2:calculate",
        "item-2:show",
        "export",
    ]
    assert not hasattr(window, "_analysis_v2pa_batch")
    assert not hasattr(window, "_analysis_v2pa_warning_callback")


def test_record_only_run_lists_one_uncalibrated_channel_once(monkeypatch):
    events = []
    resolve_calls = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(4, 4))

    def fake_resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )
    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [
        (
            window,
            "麦克风未进行校准，结果仅供参考。\n"
            "未校准通道：\n"
            "• In5",
        )
    ]
    assert resolve_calls == [4]


def test_record_only_run_uses_fresh_uncalibrated_channel_collection_each_time(
    monkeypatch,
):
    events = []
    resolve_calls = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(1,))

    def fake_resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )
    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)
    sequence_widget.SequenceWindow.run(window)

    expected_warning = (
        window,
        "麦克风未进行校准，结果仅供参考。\n"
        "未校准通道：\n"
        "• In2",
    )
    assert warning_calls == [expected_warning, expected_warning]
    assert resolve_calls == [1, 1]
    assert not hasattr(window, "_analysis_v2pa_batch")
    assert not hasattr(window, "_analysis_v2pa_warning_callback")


def test_record_only_calibrated_run_has_no_calibration_warning(monkeypatch):
    events = []
    resolve_calls = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(3,))

    def fake_resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 2.5

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )
    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == []
    assert resolve_calls == [3]
    assert events == ["item-0:calculate", "item-0:show", "export"]


def test_record_only_run_preserves_extra_calibration_diagnostic_in_same_warning(
    monkeypatch,
):
    events = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(2,))

    def fake_resolver(raw_channel, warn_callback=None):
        warn_callback("麦克风未进行校准，结果仅供参考。")
        warn_callback("校准数据诊断")
        return 1.0

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )
    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [
        (
            window,
            "麦克风未进行校准，结果仅供参考。\n"
            "未校准通道：\n"
            "• In3\n"
            "校准数据诊断",
        )
    ]


def test_record_only_exact_value_error_uses_dedicated_warning_without_channel_context(
    monkeypatch,
):
    events = []
    warning_calls = []
    window = _run_window(mode="RECORD_ONLY")
    _configure_live_batch_run(window, events, raw_channels=(0,))
    window.instance_analysis_class = (
        lambda key, item_type, config: window._analysis_v2pa_batch.resolve(
            config["analysis_channel"]
        )
    )

    def fake_resolver(raw_channel, warn_callback=None):
        raise ValueError("麦克风未进行校准，结果仅供参考。")

    def batch_factory(resolver=None):
        return AnalysisV2paBatch(resolver=resolver or fake_resolver)

    monkeypatch.setattr(
        sequence_widget,
        "resolve_analysis_v2pa_factor_for_channel",
        fake_resolver,
    )
    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "麦克风未进行校准，结果仅供参考。")]


def test_play_and_record_run_uses_dedicated_uncalibrated_warning(monkeypatch):
    events = []
    resolve_calls = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events, raw_channels=(2,))

    def fake_resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    def batch_factory():
        return AnalysisV2paBatch(resolver=fake_resolver)

    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    _patch_unsuppressed_dedicated_warning(monkeypatch, warning_calls)

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "麦克风未进行校准，结果仅供参考。")]
    assert resolve_calls == [2]


def test_live_run_batches_warnings_before_calculation_and_repeats_on_second_operation(monkeypatch):
    events = []
    resolve_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events)

    def resolver(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        warn_callback(f"In{raw_channel + 1} 未校准")
        return 1.0

    batches = []

    def batch_factory():
        batch = AnalysisV2paBatch(resolver=resolver)
        batches.append(batch)
        return batch

    warning_calls = []

    def warning(*args):
        warning_calls.append(args)
        events.append("warning")

    monkeypatch.setattr(sequence_widget, "AnalysisV2paBatch", batch_factory)
    monkeypatch.setattr(sequence_widget.MessageBox, "warning", warning)

    sequence_widget.SequenceWindow.run(window)
    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [
        (window, "提示", "• In1 未校准\n• In2 未校准"),
        (window, "提示", "• In1 未校准\n• In2 未校准"),
    ]
    expected_operation_events = [
        "warning",
        "first:calculate",
        "first:show",
        "second:calculate",
        "second:show",
        "export",
    ]
    assert events == expected_operation_events * 2
    assert resolve_calls == [0, 1, 0, 1]
    assert len(batches) == 2
    assert batches[0] is not batches[1]
    assert not hasattr(window, "_analysis_v2pa_batch")
    assert not hasattr(window, "_analysis_v2pa_warning_callback")


def test_non_target_calibration_value_error_uses_ordinary_warning(monkeypatch):
    events = []
    warning_calls = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events, raw_channels=(0,))
    window.instance_analysis_class = (
        lambda key, item_type, config: window._analysis_v2pa_batch.resolve(
            config["analysis_channel"]
        )
    )

    def resolver(raw_channel, warn_callback=None):
        raise ValueError("校准系数格式无效")

    monkeypatch.setattr(
        sequence_widget,
        "AnalysisV2paBatch",
        lambda: AnalysisV2paBatch(resolver=resolver),
    )
    monkeypatch.setattr(
        sequence_widget,
        "is_uncalibrated_microphone_warning_suppressed",
        lambda logger=None: pytest.fail("non-target diagnostics must not read preference"),
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == [(window, "提示", "校准系数格式无效")]


def test_calibrated_live_run_has_no_calibration_warning(monkeypatch):
    events = []
    window = _run_window(mode="PLAY_AND_RECORD")
    _configure_live_batch_run(window, events)
    warning_calls = []
    previous_batch = object()
    previous_callback = object()
    window._analysis_v2pa_batch = previous_batch
    window._analysis_v2pa_warning_callback = previous_callback

    monkeypatch.setattr(
        sequence_widget,
        "AnalysisV2paBatch",
        lambda: AnalysisV2paBatch(resolver=lambda raw_channel, warn_callback=None: raw_channel + 1.0),
    )
    monkeypatch.setattr(
        sequence_widget.MessageBox,
        "warning",
        lambda *args: warning_calls.append(args),
    )

    sequence_widget.SequenceWindow.run(window)

    assert warning_calls == []
    assert events == [
        "first:calculate",
        "first:show",
        "second:calculate",
        "second:show",
        "export",
    ]
    assert window._analysis_v2pa_batch is previous_batch
    assert window._analysis_v2pa_warning_callback is previous_callback
