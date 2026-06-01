import ast
import re
import textwrap
import types
from pathlib import Path

import numpy as np


SOURCE_PATH = Path(__file__).resolve().parents[2] / "ui" / "sequence" / "sequence_widget.py"
TARGET_METHODS = {
    "_get_mode_display_name",
    "_clear_sn_for_external_trigger_rejection",
    "_ensure_external_trigger_mode_supported",
    "_normalize_barcode",
    "_barcode_has_invalid_chars",
    "_reset_barcode_commit_state",
    "_validate_sn_regex_before_start",
    "_tcp_run_test",
    "_commit_barcode",
    "_set_sn_input_recording_read_only",
    "on_sequence_config_updated",
    "validate_count",
    "start_this_play",
    "judge_play_and_record",
    "_should_use_streaming_recording",
    "_start_streaming_recording",
    "_normalize_blocking_recorded_data",
    "_finish_recording_success",
    "_finish_recording_failure",
    "_start_blocking_recording",
    "_on_streaming_complete",
    "eventFilter",
}


def _build_method_namespace():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sequence_window_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )

    method_sources = {}
    for node in sequence_window_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_METHODS:
            method_sources[node.name] = textwrap.dedent(ast.get_source_segment(source, node))

    class DummyApplication:
        @staticmethod
        def processEvents():
            return None

        @staticmethod
        def focusWidget():
            return None

    class DummySignalBlocker:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMessageBox:
        warnings = []

        @staticmethod
        def warning(*args, **kwargs):
            DummyMessageBox.warnings.append((args, kwargs))

    class DummyLoadUiConfig:
        @staticmethod
        def load_recorded_num_from_json(logger):
            return 1, "SN001"

    class DummyEventType:
        MouseButtonPress = "mouse_press"
        KeyPress = "key_press"
        Move = "move"
        Resize = "resize"

    class DummyLineEdit:
        pass

    class DummyQt:
        ControlModifier = 0x04000000
        Key_Backspace = "backspace"
        Key_Delete = "delete"
        Key_Left = "left"
        Key_Right = "right"
        Key_Up = "up"
        Key_Down = "down"
        Key_Home = "home"
        Key_End = "end"
        Key_PageUp = "page_up"
        Key_PageDown = "page_down"
        Key_Z = "z"

    namespace = {
        "QApplication": DummyApplication,
        "QSignalBlocker": DummySignalBlocker,
        "pyqtSlot": lambda *args, **kwargs: (lambda func: func),
        "MessageBox": DummyMessageBox,
        "LoadUiConfig": DummyLoadUiConfig,
        "QEvent": DummyEventType,
        "Qt": DummyQt,
        "QLineEdit": DummyLineEdit,
        "StreamingWavWriter": None,
        "stream_play_and_record": None,
        "stream_record_without_play": None,
        "play_last_stimulus_wave": None,
        "record_without_play": None,
        "RecordingManager": None,
        "AlignmentProcessing": types.SimpleNamespace(
            align_play_and_rec_data_using_gccphat=lambda stimulus, recorded: recorded
        ),
        "SplitRepeatSignal": lambda: types.SimpleNamespace(split_repeat_signal=lambda data, sample_rate, **kwargs: []),
        "error_code": types.SimpleNamespace(OK="OK"),
        "save_audio_simple": lambda *args, **kwargs: None,
        "save_recorded_data_to_json": lambda *args, **kwargs: None,
        "TempTcpClient": lambda *args, **kwargs: None,
        "np": np,
        "SoundcardAudioProcessor": None,
        "re": re,
        "time": types.SimpleNamespace(monotonic=lambda: 1000.0),
        "os": types.SimpleNamespace(
            path=types.SimpleNamespace(exists=lambda path: False),
            remove=lambda path: None,
        ),
    }

    for method_name in TARGET_METHODS:
        exec(method_sources[method_name], namespace)

    return namespace


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))

    def debug(self, message):
        self.messages.append(("debug", message))


class FakeLineEdit:
    def __init__(self, text="SN001"):
        self._text = text
        self.read_only = False
        self.focus_calls = 0
        self.select_all_calls = 0
        self.set_text_calls = []
        self.clear_calls = 0
        self._selected_text = ""

    def setReadOnly(self, value):
        self.read_only = value

    def isReadOnly(self):
        return self.read_only

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self._selected_text = ""
        self.set_text_calls.append(value)

    def clear(self):
        self._text = ""
        self._selected_text = ""
        self.clear_calls += 1

    def setFocus(self):
        self.focus_calls += 1

    def selectAll(self):
        self.select_all_calls += 1
        self._selected_text = self._text

    def selectedText(self):
        return self._selected_text

    def setSelectedText(self, value):
        self._selected_text = value


class FakeMouseEvent:
    def __init__(self, event_type):
        self._event_type = event_type

    def type(self):
        return self._event_type


class FakeKeyEvent:
    def __init__(self, event_type, key, text="", modifiers=0):
        self._event_type = event_type
        self._key = key
        self._text = text
        self._modifiers = modifiers

    def type(self):
        return self._event_type

    def key(self):
        return self._key

    def text(self):
        return self._text

    def modifiers(self):
        return self._modifiers


class FakeButton:
    def __init__(self):
        self.disabled = None
        self.enabled = None

    def setDisabled(self, value):
        self.disabled = value

    def setEnabled(self, value):
        self.enabled = value


class FakeCheckBox:
    def __init__(self, checked=True, enabled=True):
        self.checked = checked
        self.enabled = enabled

    def isChecked(self):
        return self.checked

    def isEnabled(self):
        return self.enabled

    def setEnabled(self, value):
        self.enabled = value


class FakeGraph:
    def __init__(self):
        self.clear_calls = 0
        self.plot_calls = []

    def clear(self):
        self.clear_calls += 1

    def plot(self, *args, **kwargs):
        plot_item = FakePlotItem()
        self.plot_calls.append((args, kwargs, plot_item))
        return plot_item


class FakePlotItem:
    def __init__(self):
        self.data_calls = []

    def setData(self, *args):
        self.data_calls.append(args)


class FakeTimer:
    def __init__(self):
        self.started_intervals = []
        self.active = False

    def start(self, interval):
        self.started_intervals.append(interval)
        self.active = True

    def stop(self):
        self.active = False

    def isActive(self):
        return self.active


class FakeStreamingProcessor:
    def __init__(self, recorded_data=None, recorded_data_multi=None, raise_on_get=None):
        self.recorded_data = list(recorded_data or [0.1, 0.2, 0.3])
        self.recorded_data_multi = recorded_data_multi
        self.raise_on_get = raise_on_get
        self.is_recording = True
        self.target_samples = len(self.recorded_data)

    def process_queue(self):
        return None

    def get_recorded_data(self):
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return self.recorded_data

    def get_recorded_data_multi(self):
        if self.recorded_data_multi is None:
            return None
        return self.recorded_data_multi

    def stop_streaming(self):
        self.is_recording = False


class FakeWavWriter:
    def __init__(self, path, sample_rate, **kwargs):
        self.path = path
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        self.finalized = False

    def finalize(self):
        self.finalized = True


def _bind_method(obj, namespace, method_name):
    return namespace[method_name].__get__(obj, type(obj))


def _build_fake_window(namespace, *, use_streaming=True, mode="RECORD_ONLY", reset_result=None):
    if reset_result is None:
        reset_result = ({"stimulus": True}, {"recorded": True}, 48000)

    class FakeWindow:
        pass

    window = FakeWindow()
    window.default_logger = FakeLogger()
    window.lineedit_s_or_n = FakeLineEdit()
    window.lineedit_count = FakeLineEdit("1")
    window.lineedit_type = FakeLineEdit("MODEL")
    window.analysis_window = []
    window._analysis_result_summary_window = None
    window._analysis_window_key_by_obj = {}
    window._record_workflow_busy = False
    window._barcode_scanner_box_enabled_before_recording = None
    window.tcp_flag = False
    window.player_status_flag = False
    window.clicked_player_flag = False
    window.current_recorded_count = 1
    window.last_play_count = 1
    acq_detail = {}
    if use_streaming is not None:
        acq_detail["use_streaming_recording"] = bool(use_streaming)
    window.sequence_config = [{"seq1": {"acq": {"mode": mode, "detail": acq_detail}}}]
    window.mode = mode
    window.use_streaming = use_streaming
    window.line_graph = FakeGraph()
    window.replayer_btn = FakeButton()
    window.data_btn = FakeButton()
    window.streaming_buffer = []
    window.streaming_time_data = []
    window.streaming_plot_item = None
    window.streaming_wav_writer = None
    window.streaming_processor = None
    window.streaming_stimulus_data = None
    window.streaming_mode = None
    window.streaming_poll_timer = FakeTimer()
    window.recorded_path = "demo.wav"
    window.recorded_signal_info = {}
    window.analysis_config = {"auto_analysis": False}
    window.data_struct = types.SimpleNamespace(
        sample_rate=48000,
        stimulus_info={"repeat_times": 1},
        store_wave_data=[0.1, 0.2, 0.3],
        store_wave_data_multi=np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
        split_repeat_data=None,
    )
    window.barcode_scanner_box = FakeCheckBox()
    window._awaiting_ok_ng = False
    window._sn_clear_on_next_scan = False
    window._sn_textchange_manual_guard = False
    window._INVALID_FILENAME_CHARS = set('\\/:*?"<>|')
    window._barcode_first_char_ts = 1.0
    window._barcode_last_char_ts = 2.0
    window._barcode_capture_buffer = "PENDING"
    window._barcode_capture_first_ts = 3.0
    window._barcode_capture_last_ts = 4.0
    window._barcode_capture_target_lineedit = object()
    window._barcode_capture_target_text = "PENDING"
    window._barcode_capture_target_cursor_pos = 5
    window._barcode_debounce_timer = FakeTimer()
    window._barcode_debounce_timer.active = True
    window._last_committed_barcode = None
    window._last_committed_barcode_time = 0.0
    window._barcode_commit_dedup_window_sec = 0.8
    window._hid_mode_active_until = 0.0
    window.run_called = False
    window.run_invocations = []
    window.start_calls = []
    window.close_analysis_calls = 0
    window.paused_updates = 0
    window.playing_updates = 0
    window.cleanup_calls = 0
    window.checked_work_status_message = lambda: False
    window.reset_work_pram = lambda label, count=None: reset_result
    window._cleanup_streaming_resources = lambda: setattr(window, "cleanup_calls", window.cleanup_calls + 1)
    window._clear_plot_area = lambda: window.line_graph.clear()
    window.update_player_btn_is_playing = lambda: setattr(window, "playing_updates", window.playing_updates + 1)
    window.update_player_btn_is_paused = lambda: setattr(window, "paused_updates", window.paused_updates + 1)
    window.plot_line_graph = lambda *args, **kwargs: setattr(window, "plot_called", True)
    window.plot_calls = []
    window.plot_waveform_to_workspace = lambda *args, **kwargs: window.plot_calls.append((args, kwargs))
    def _run_stub():
        window.run_called = True
        window.run_invocations.append(
            {
                "busy": window._record_workflow_busy,
                "read_only": window.lineedit_s_or_n.isReadOnly(),
                "player_status_flag": window.player_status_flag,
            }
        )
    window.run = _run_stub
    window._close_analysis_windows = lambda: setattr(window, "close_analysis_calls", window.close_analysis_calls + 1)
    window.start_this_play = lambda label, skip_sn_regex_validation=False: window.start_calls.append(label)
    window._load_selected_sn_regex_rule = lambda: {"name": "default", "pattern": r"SN\d+"}
    window._normalize_barcode = _bind_method(window, namespace, "_normalize_barcode")
    window._barcode_has_invalid_chars = _bind_method(window, namespace, "_barcode_has_invalid_chars")
    window._reset_barcode_commit_state = _bind_method(window, namespace, "_reset_barcode_commit_state")
    window._validate_sn_regex_before_start = _bind_method(window, namespace, "_validate_sn_regex_before_start")
    window._tcp_run_test = _bind_method(window, namespace, "_tcp_run_test")
    window._get_mode_display_name = _bind_method(window, namespace, "_get_mode_display_name")
    window._clear_sn_for_external_trigger_rejection = _bind_method(
        window, namespace, "_clear_sn_for_external_trigger_rejection"
    )
    window._show_external_trigger_mode_warning = lambda trigger_source, current_mode: namespace["MessageBox"].warning(
        window,
        "提示",
        f"{current_mode} 不支持{trigger_source}启动工作流",
    )
    window._ensure_external_trigger_mode_supported = _bind_method(
        window, namespace, "_ensure_external_trigger_mode_supported"
    )
    window._commit_barcode = _bind_method(window, namespace, "_commit_barcode")
    window._set_sn_input_recording_read_only = _bind_method(window, namespace, "_set_sn_input_recording_read_only")
    window.on_sequence_config_updated = _bind_method(window, namespace, "on_sequence_config_updated")
    window.validate_count = _bind_method(window, namespace, "validate_count")
    window._real_start_this_play = _bind_method(window, namespace, "start_this_play")
    window.judge_play_and_record = _bind_method(window, namespace, "judge_play_and_record")
    window._should_use_streaming_recording = _bind_method(window, namespace, "_should_use_streaming_recording")
    window._start_streaming_recording = _bind_method(window, namespace, "_start_streaming_recording")
    window._normalize_blocking_recorded_data = _bind_method(window, namespace, "_normalize_blocking_recorded_data")
    window._finish_recording_success = _bind_method(window, namespace, "_finish_recording_success")
    window._finish_recording_failure = _bind_method(window, namespace, "_finish_recording_failure")
    window._start_blocking_recording = _bind_method(window, namespace, "_start_blocking_recording")
    window._on_streaming_complete = _bind_method(window, namespace, "_on_streaming_complete")
    window.eventFilter = _bind_method(window, namespace, "eventFilter")
    return window


def test_streaming_mode_keeps_sn_locked_until_completion():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_play_and_record"] = lambda *args, **kwargs: (FakeStreamingProcessor(), [1, 2, 3], None)
    namespace["stream_record_without_play"] = lambda *args, **kwargs: (FakeStreamingProcessor(), None)
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")

    window.judge_play_and_record()

    assert window.lineedit_s_or_n.isReadOnly() is True
    assert window.barcode_scanner_box.isEnabled() is False
    assert window.streaming_poll_timer.started_intervals == [50]
    assert window._record_workflow_busy is True

    window._on_streaming_complete()

    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.barcode_scanner_box.isEnabled() is True
    assert window._record_workflow_busy is False
    assert window.data_btn.enabled is True
    assert window.replayer_btn.enabled is True
    assert window.lineedit_s_or_n.focus_calls == 1
    assert window.lineedit_s_or_n.select_all_calls == 1


def test_record_only_streaming_completion_stores_mono_and_multi_recorded_data():
    namespace = _build_method_namespace()
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )
    mono_data = [0.1, 0.2, 0.3]
    multi_data = np.array(
        [
            [0.1, 1.1],
            [0.2, 1.2],
            [0.3, 1.3],
        ],
        dtype=np.float32,
    )

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.streaming_mode = "record_only"
    window.streaming_processor = FakeStreamingProcessor(
        recorded_data=mono_data,
        recorded_data_multi=multi_data,
    )

    window._on_streaming_complete()

    np.testing.assert_array_equal(
        window.data_struct.store_wave_data,
        multi_data.mean(axis=1).astype(np.float32, copy=False),
    )
    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, multi_data)


def test_streaming_start_failure_restores_sn_editability():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_play_and_record"] = lambda *args, **kwargs: (FakeStreamingProcessor(), [1, 2, 3], None)

    def _raise_start_error(*args, **kwargs):
        raise RuntimeError("stream start failed")

    namespace["stream_record_without_play"] = _raise_start_error

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")

    window.judge_play_and_record()

    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.barcode_scanner_box.isEnabled() is True
    assert window._record_workflow_busy is False
    assert window.player_status_flag is False
    assert window.paused_updates == 1


def test_missing_streaming_flag_uses_blocking_recording():
    namespace = _build_method_namespace()
    calls = {"stream": 0, "blocking": 0}
    namespace["stream_record_without_play"] = lambda *args, **kwargs: calls.__setitem__(
        "stream", calls["stream"] + 1
    )

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            calls["blocking"] += 1
            return namespace["error_code"].OK, np.array([1.0, 2.0], dtype=np.float32)

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )
    namespace["save_audio_simple"] = lambda *args, **kwargs: None

    window = _build_fake_window(namespace, use_streaming=None, mode="RECORD_ONLY")
    window.sequence_config[0]["seq1"]["acq"]["detail"].pop("use_streaming_recording", None)

    window.judge_play_and_record()

    assert calls == {"stream": 0, "blocking": 1}
    assert window.data_struct.store_wave_data.tolist() == [1.0, 2.0]
    assert window._record_workflow_busy is False


def test_streaming_flag_uses_streaming_recording():
    namespace = _build_method_namespace()
    calls = {"stream": 0, "blocking": 0}
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_record_without_play"] = lambda *args, **kwargs: (
        calls.__setitem__("stream", calls["stream"] + 1) or FakeStreamingProcessor(),
        None,
    )

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            calls["blocking"] += 1
            return namespace["error_code"].OK, np.array([1.0, 2.0], dtype=np.float32)

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")

    window.judge_play_and_record()

    assert calls == {"stream": 1, "blocking": 0}
    assert window._record_workflow_busy is True


def test_play_record_missing_streaming_flag_uses_blocking_recording():
    namespace = _build_method_namespace()
    calls = {"playrec": 0, "stream": 0}
    namespace["stream_play_and_record"] = lambda *args, **kwargs: calls.__setitem__(
        "stream", calls["stream"] + 1
    )

    class FakeSoundcardAudioProcessor:
        def sd_play_rec(self, recorded_dict, stimulus_dict, path):
            calls["playrec"] += 1
            return namespace["error_code"].OK, np.array([1.0, 2.0], dtype=np.float32)

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )
    window = _build_fake_window(namespace, use_streaming=None, mode="PLAY_AND_RECORD")
    window.sequence_config[0]["seq1"]["acq"]["detail"].pop("use_streaming_recording", None)

    window.judge_play_and_record()

    assert calls == {"playrec": 1, "stream": 0}
    assert window.data_struct.store_wave_data.tolist() == [1.0, 2.0]


def test_reset_failure_and_empty_stimulus_restore_sn_editability():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_play_and_record"] = lambda *args, **kwargs: (FakeStreamingProcessor(), [1, 2, 3], None)
    namespace["stream_record_without_play"] = lambda *args, **kwargs: (FakeStreamingProcessor(), None)

    reset_error_window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")

    def _raise_reset_error(label, count=None):
        raise RuntimeError("reset failed")

    reset_error_window.reset_work_pram = _raise_reset_error
    reset_error_window.judge_play_and_record()

    assert reset_error_window.lineedit_s_or_n.isReadOnly() is False
    assert reset_error_window.barcode_scanner_box.isEnabled() is True
    assert reset_error_window._record_workflow_busy is False

    empty_stimulus_window = _build_fake_window(
        namespace,
        use_streaming=True,
        mode="RECORD_ONLY",
        reset_result=(None, None, None),
    )
    empty_stimulus_window.judge_play_and_record()

    assert empty_stimulus_window.lineedit_s_or_n.isReadOnly() is False
    assert empty_stimulus_window.barcode_scanner_box.isEnabled() is True
    assert empty_stimulus_window._record_workflow_busy is False


def test_streaming_completion_error_restores_sn_editability():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window._record_workflow_busy = True
    window.player_status_flag = True
    window.streaming_mode = "record_only"
    window.streaming_processor = FakeStreamingProcessor(raise_on_get=RuntimeError("completion failed"))
    window.streaming_wav_writer = FakeWavWriter("demo.wav", 48000)

    window._on_streaming_complete()

    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.barcode_scanner_box.isEnabled() is True
    assert window._record_workflow_busy is False
    assert window.player_status_flag is False
    assert window.data_btn.enabled is True
    assert window.replayer_btn.enabled is True


def test_blocking_success_and_failure_restore_sn_editability():
    namespace = _build_method_namespace()
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    class SuccessSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            return namespace["error_code"].OK, np.array([1.0, 2.0], dtype=np.float32)

    namespace["SoundcardAudioProcessor"] = SuccessSoundcardAudioProcessor

    success_window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")
    success_window.judge_play_and_record()

    assert success_window.lineedit_s_or_n.isReadOnly() is False
    assert success_window.barcode_scanner_box.isEnabled() is True
    assert success_window._record_workflow_busy is False
    assert success_window._awaiting_ok_ng is True
    assert success_window._sn_clear_on_next_scan is True

    class FailingSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            raise RuntimeError("blocking failed")

    namespace["SoundcardAudioProcessor"] = FailingSoundcardAudioProcessor
    failure_window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")
    failure_window.judge_play_and_record()

    assert failure_window.lineedit_s_or_n.isReadOnly() is False
    assert failure_window.barcode_scanner_box.isEnabled() is True
    assert failure_window._record_workflow_busy is False
    assert failure_window.player_status_flag is False
    assert failure_window._awaiting_ok_ng is False
    assert failure_window._sn_clear_on_next_scan is False


def test_blocking_processes_events_before_hardware_call():
    namespace = _build_method_namespace()
    events = []

    class FakeApplication:
        @staticmethod
        def processEvents():
            events.append("process_events")

        @staticmethod
        def focusWidget():
            return None

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            events.append("hardware_call")
            return namespace["error_code"].OK, np.array([1.0], dtype=np.float32)

    namespace["QApplication"] = FakeApplication
    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )

    window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")

    window.judge_play_and_record()

    assert events == ["process_events", "hardware_call"]


def test_blocking_failure_does_not_save_or_run_analysis():
    namespace = _build_method_namespace()
    calls = {"save_db": 0}

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            return "INVALID_RECORD", None

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: calls.__setitem__("save_db", calls["save_db"] + 1)
    )

    window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")
    window.analysis_config = {"auto_analysis": True}

    window.judge_play_and_record()

    assert calls["save_db"] == 0
    assert window.run_called is False
    assert window._record_workflow_busy is False
    assert window.player_status_flag is False
    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.data_btn.enabled is True
    assert window.replayer_btn.enabled is True


def test_blocking_record_only_multi_channel_stores_multi_and_mean_mono():
    namespace = _build_method_namespace()
    recorded = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            return namespace["error_code"].OK, recorded

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )
    saved_audio = []
    namespace["save_audio_simple"] = lambda path, data, sample_rate: saved_audio.append(
        (path, np.asarray(data), sample_rate)
    )

    window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")
    window._active_input_channels = [0, 1]

    window.judge_play_and_record()

    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, recorded)
    np.testing.assert_array_equal(window.data_struct.store_wave_data, np.array([2.0, 3.0], dtype=np.float32))
    assert window.data_struct.store_wave_data_multi.shape == (2, 2)
    np.testing.assert_array_equal(saved_audio[0][1], recorded)
    assert window.plot_calls[0][0][0].shape == (2, 2)


def test_blocking_play_record_normalizes_frames_by_channels_output():
    namespace = _build_method_namespace()
    recorded = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    class FakeSoundcardAudioProcessor:
        def sd_play_rec(self, recorded_dict, stimulus_dict, path):
            return namespace["error_code"].OK, recorded

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )

    window = _build_fake_window(namespace, use_streaming=False, mode="PLAY_AND_RECORD")
    window._active_input_channels = [0, 1]

    window.judge_play_and_record()

    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, recorded)
    np.testing.assert_array_equal(window.data_struct.store_wave_data, np.array([2.0, 3.0], dtype=np.float32))
    assert window.data_struct.store_wave_data_multi.shape == (2, 2)


def test_blocking_play_record_transposes_channel_by_frames_output():
    namespace = _build_method_namespace()
    recorded_transposed = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]], dtype=np.float32)

    class FakeSoundcardAudioProcessor:
        def sd_play_rec(self, recorded_dict, stimulus_dict, path):
            return namespace["error_code"].OK, recorded_transposed

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: (namespace["error_code"].OK, "saved")
    )

    window = _build_fake_window(namespace, use_streaming=False, mode="PLAY_AND_RECORD")
    window._active_input_channels = [0, 1]

    window.judge_play_and_record()

    expected_multi = recorded_transposed.T
    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, expected_multi)
    np.testing.assert_array_equal(
        window.data_struct.store_wave_data,
        expected_multi.mean(axis=1).astype(np.float32, copy=False),
    )
    assert window.data_struct.store_wave_data_multi.shape == (3, 2)


def test_sn_lock_restores_previously_disabled_scanner_checkbox_state():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.barcode_scanner_box = FakeCheckBox(enabled=False)

    window._set_sn_input_recording_read_only(True)
    assert window.barcode_scanner_box.isEnabled() is False

    window._set_sn_input_recording_read_only(False)
    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.barcode_scanner_box.isEnabled() is False


def test_blocking_auto_analysis_releases_sn_lock_before_run():
    namespace = _build_method_namespace()
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            return namespace["error_code"].OK, np.array([1.0, 2.0], dtype=np.float32)

    namespace["SoundcardAudioProcessor"] = FakeSoundcardAudioProcessor

    window = _build_fake_window(namespace, use_streaming=False, mode="RECORD_ONLY")
    window.analysis_config = {"auto_analysis": True}
    events = []

    original_set_read_only = window._set_sn_input_recording_read_only

    def _tracked_set_read_only(value):
        original_set_read_only(value)
        events.append(("set_read_only", value, window._record_workflow_busy))

    def _tracked_run():
        events.append(("run", window._record_workflow_busy, window.lineedit_s_or_n.isReadOnly()))
        window.run_called = True
        window.run_invocations.append(
            {
                "busy": window._record_workflow_busy,
                "read_only": window.lineedit_s_or_n.isReadOnly(),
                "player_status_flag": window.player_status_flag,
            }
        )

    window._set_sn_input_recording_read_only = _tracked_set_read_only
    window.run = _tracked_run

    window.judge_play_and_record()

    assert window.run_called is True
    assert window.run_invocations == [
        {"busy": False, "read_only": False, "player_status_flag": False}
    ]
    assert events[-2:] == [("set_read_only", False, False), ("run", False, False)]
    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window._record_workflow_busy is False


def test_streaming_auto_analysis_releases_sn_lock_before_run():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_play_and_record"] = lambda *args, **kwargs: (FakeStreamingProcessor(), [1, 2, 3], None)
    namespace["stream_record_without_play"] = lambda *args, **kwargs: (FakeStreamingProcessor(), None)
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.analysis_config = {"auto_analysis": True}
    events = []

    original_set_read_only = window._set_sn_input_recording_read_only

    def _tracked_set_read_only(value):
        original_set_read_only(value)
        events.append(("set_read_only", value, window._record_workflow_busy))

    def _tracked_run():
        events.append(("run", window._record_workflow_busy, window.lineedit_s_or_n.isReadOnly()))
        window.run_called = True
        window.run_invocations.append(
            {
                "busy": window._record_workflow_busy,
                "read_only": window.lineedit_s_or_n.isReadOnly(),
                "player_status_flag": window.player_status_flag,
            }
        )

    window._set_sn_input_recording_read_only = _tracked_set_read_only
    window.run = _tracked_run

    window.judge_play_and_record()
    window._on_streaming_complete()

    assert window.run_called is True
    assert window.run_invocations == [
        {"busy": False, "read_only": False, "player_status_flag": False}
    ]
    assert events[-2:] == [("set_read_only", False, False), ("run", False, False)]
    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window._record_workflow_busy is False


def test_streaming_auto_analysis_unlock_event_happens_before_run():
    namespace = _build_method_namespace()
    namespace["StreamingWavWriter"] = FakeWavWriter
    namespace["stream_play_and_record"] = lambda *args, **kwargs: (FakeStreamingProcessor(), [1, 2, 3], None)
    namespace["stream_record_without_play"] = lambda *args, **kwargs: (FakeStreamingProcessor(), None)
    namespace["RecordingManager"] = lambda: types.SimpleNamespace(
        save_signal_info_to_db=lambda *args, **kwargs: ("OK", "saved")
    )

    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.analysis_config = {"auto_analysis": True}
    events = []

    original_set_read_only = window._set_sn_input_recording_read_only

    def _tracked_set_read_only(value):
        original_set_read_only(value)
        if value is False:
            events.append(("unlock", window._record_workflow_busy, window.lineedit_s_or_n.isReadOnly()))

    def _tracked_run():
        events.append(("run", window._record_workflow_busy, window.lineedit_s_or_n.isReadOnly()))
        window.run_called = True

    window._set_sn_input_recording_read_only = _tracked_set_read_only
    window.run = _tracked_run

    window.judge_play_and_record()
    window._on_streaming_complete()

    assert events == [("unlock", False, False), ("run", False, False)]
    assert window.run_called is True


def test_sequence_config_update_keeps_reloaded_play_record_sample_rate():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="PLAY_AND_RECORD")
    window.data_struct.sample_rate = 44100
    window.sequence_config = [
        {
            "seq1": {
                "acq": {
                    "mode": "PLAY_AND_RECORD",
                    "detail": {
                        "stimulus_info": {
                            "sample_rate": 48000,
                            "total_time": 4.0,
                            "amplitude": 1.0,
                        },
                    },
                },
            },
        }
    ]
    window.count_board = None
    window.update_using_file_combobox = lambda: None
    window.get_sequence_config_from_json = lambda: None
    window.init_fft_and_stft_flag = lambda: None
    window.refresh_channel_windows = lambda: None
    window._refresh_test_mode_availability = lambda: None

    def _reload_stimulus_config():
        window.data_struct.sample_rate = window.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"][
            "sample_rate"
        ]

    window.init_data_struct_stimulus_config = _reload_stimulus_config

    window.on_sequence_config_updated()

    assert window.data_struct.sample_rate == 48000


def test_commit_barcode_ignores_busy_scan_without_overwriting_sn():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window._record_workflow_busy = True
    window.player_status_flag = True

    window._commit_barcode("SN-NEW", source="hid")

    assert window.lineedit_s_or_n.text() == "SN-OLD"
    assert window.lineedit_s_or_n.set_text_calls == []
    assert window.start_calls == []
    assert window._last_committed_barcode is None
    assert window._last_committed_barcode_time == 0.0
    assert window._barcode_debounce_timer.isActive() is False
    assert window._barcode_first_char_ts is None
    assert window._barcode_last_char_ts is None
    assert window._barcode_capture_buffer == ""
    assert window._barcode_capture_first_ts is None
    assert window._barcode_capture_last_ts is None
    assert window._barcode_capture_target_lineedit is None
    assert window._barcode_capture_target_text is None
    assert window._barcode_capture_target_cursor_pos is None
    assert any("忽略扫码提交" in message for _, message in window.default_logger.messages)


def test_commit_barcode_when_idle_still_updates_sn_and_starts_test():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}

    window._commit_barcode("SN-123", source="hid")

    assert window.lineedit_s_or_n.text() == "SN-123"
    assert window.lineedit_s_or_n.set_text_calls == ["SN-123"]
    assert window.start_calls == ["not_labeled"]
    assert window.close_analysis_calls == 1
    assert window._last_committed_barcode == "SN-123"
    assert window._last_committed_barcode_time == 1000.0


def test_start_this_play_blocks_invalid_sn_for_non_scan_entry():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.start_this_play = window._real_start_this_play
    window.lineedit_s_or_n = FakeLineEdit("BAD-SN")
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}
    judge_calls = []
    window.judge_play_and_record = lambda label, is_replay=False: judge_calls.append((label, is_replay))

    window.start_this_play("not_labeled")

    assert judge_calls == []
    assert window.current_recorded_count == 1
    assert window.lineedit_count.text() == "1"
    warnings = namespace["MessageBox"].warnings
    assert len(warnings) == 1
    assert "规则名称：sn-rule" in warnings[0][0][2]
    assert "规则表达式：SN-\\d{3}" in warnings[0][0][2]
    assert "实际 SN 内容：BAD-SN" in warnings[0][0][2]


def test_tcp_run_test_allows_invalid_sn_when_validation_is_explicitly_skipped():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.start_this_play = window._real_start_this_play
    window.lineedit_s_or_n = FakeLineEdit("BAD-SN")
    window.barcode_scanner_box = FakeCheckBox(checked=True)
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}
    window.tcp_flag = True
    judge_calls = []
    window.judge_play_and_record = lambda label, is_replay=False: judge_calls.append((label, is_replay))

    window._tcp_run_test("not_labeled", skip_sn_regex_validation=True)

    assert judge_calls == [("not_labeled", False)]
    assert window.current_recorded_count == 2
    assert window.lineedit_count.text() == "2"
    assert namespace["MessageBox"].warnings == []


def test_tcp_run_test_allows_play_and_record_mode():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="PLAY_AND_RECORD")
    window.start_this_play = lambda label, skip_sn_regex_validation=False: window.start_calls.append(
        (label, skip_sn_regex_validation)
    )

    window._tcp_run_test("OK", skip_sn_regex_validation=True)

    assert window.start_calls == [("OK", True)]
    assert namespace["MessageBox"].warnings == []


def test_tcp_run_test_blocks_unsupported_mode_with_friendly_message():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="IMPORT_AUDIO")

    window._tcp_run_test("OK", skip_sn_regex_validation=True)

    assert window.start_calls == []
    warnings = namespace["MessageBox"].warnings
    assert len(warnings) == 1
    assert "导入音频" in warnings[0][0][2]
    assert "不支持TCP启动工作流" in warnings[0][0][2]


def test_replay_path_blocks_invalid_sn_before_recording():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("BAD-SN")
    window.last_play_count = 3
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}

    window.judge_play_and_record(is_replay=True)

    assert window._record_workflow_busy is False
    assert window.player_status_flag is False
    assert window.cleanup_calls == 0
    warnings = namespace["MessageBox"].warnings
    assert len(warnings) == 1
    assert "实际 SN 内容：BAD-SN" in warnings[0][0][2]


def test_busy_count_path_keeps_sn_locked():
    namespace = _build_method_namespace()
    namespace["QLineEdit"] = FakeLineEdit
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-LOCKED")
    window.lineedit_count = FakeLineEdit("12")
    window.lineedit_count.setReadOnly(True)
    window._record_workflow_busy = True
    window.player_status_flag = True

    handled = window.eventFilter(window.lineedit_count, FakeMouseEvent(namespace["QEvent"].MouseButtonPress))
    window.validate_count(window.lineedit_count, True)

    assert handled is True
    assert window.lineedit_count.isReadOnly() is True
    assert window.lineedit_count.focus_calls == 0
    assert window.lineedit_count.select_all_calls == 0
    assert window.lineedit_s_or_n.text() == "SN-LOCKED"


def test_idle_count_path_still_unlocks_and_clears_sn():
    namespace = _build_method_namespace()
    namespace["QLineEdit"] = FakeLineEdit
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window.lineedit_count = FakeLineEdit("12")
    window.lineedit_count.setReadOnly(True)

    handled = window.eventFilter(window.lineedit_count, FakeMouseEvent(namespace["QEvent"].MouseButtonPress))
    window.validate_count(window.lineedit_count, True)

    assert handled is True
    assert window.lineedit_count.isReadOnly() is False
    assert window.lineedit_count.focus_calls == 1
    assert window.lineedit_count.select_all_calls == 1
    assert window.lineedit_s_or_n.text() == ""


def test_sn_backspace_marks_manual_edit_guard_and_resets_barcode_state():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-1234567")
    window._barcode_first_char_ts = 11.0
    window._barcode_last_char_ts = 11.2
    window._barcode_debounce_timer.active = True

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    try:
        window.eventFilter(
            window.lineedit_s_or_n,
            FakeKeyEvent(namespace["QEvent"].KeyPress, namespace["Qt"].Key_Backspace),
        )
    except RuntimeError as exc:
        assert "__class__ cell not found" in str(exc)

    assert window._sn_textchange_manual_guard is True
    assert window._barcode_first_char_ts is None
    assert window._barcode_last_char_ts is None
    assert window._barcode_debounce_timer.isActive() is False


def test_sn_full_selection_rearms_textchange_auto_commit():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window.lineedit_s_or_n.setSelectedText("SN-OLD")
    window._sn_textchange_manual_guard = True

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    try:
        window.eventFilter(
            window.lineedit_s_or_n,
            FakeKeyEvent(namespace["QEvent"].KeyPress, "S", text="S"),
        )
    except RuntimeError as exc:
        assert "__class__ cell not found" in str(exc)

    assert window._sn_textchange_manual_guard is False


def test_sn_ctrl_z_is_swallowed_without_breaking_startup_logic():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    handled = window.eventFilter(
        window.lineedit_s_or_n,
        FakeKeyEvent(
            namespace["QEvent"].KeyPress,
            namespace["Qt"].Key_Z,
            text="z",
            modifiers=namespace["Qt"].ControlModifier,
        ),
    )

    assert handled is True
