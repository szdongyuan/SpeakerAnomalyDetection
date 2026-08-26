import ast
import json
import re
import sys
import textwrap
import types
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from datetime import datetime
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pytest
from PyQt5.QtCore import QEvent as QtEvent, Qt as QtCore
from PyQt5.QtWidgets import QPushButton

try:
    import base.analysis_warning_preferences  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "base.analysis_warning_preferences":
        raise
    warning_preferences = types.ModuleType("base.analysis_warning_preferences")
    warning_preferences.is_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: False
    )
    warning_preferences.save_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: None
    )
    sys.modules[warning_preferences.__name__] = warning_preferences

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.load_config import LoadUiConfig
from base.acquisition_recording_defaults import (
    normalize_play_record_detail,
    normalize_record_only_detail,
)
from base.pre_processing.alignment_processing import AlignmentProcessing
from consts.audio_consts import normalize_float_bit_depth
from ui.sequence.barcode_router import BarcodeRouter
from ui.sequence.sequence_trigger_controller import SequenceTriggerController
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_analysis_controller import (
    SequenceAnalysisController,
    SequenceAnalysisTransportController,
)
from ui.sequence.sequence_legacy_recording_bridge import (
    LegacyRecordingAdmissionBridge,
    legacy_recording_session_snapshot,
)
from ui.sequence.sequence_messages import (
    ConfigurationSnapshot,
    ReplayRequested,
)
from ui.sequence.sequence_recording_service import (
    RecordingAdmissionInputs,
    RecordingAdmissionService,
)
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase


SOURCE_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"
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
    "set_audio_devices_available",
    "_summarize_ok_ng",
    "_ensure_pending_recorded_count",
    "_clear_pending_recorded_count",
    "_commit_pending_recorded_count",
    "_begin_recording_output_attempt",
    "_recording_output_path",
    "_clear_recording_output_attempt",
    "_finalize_successful_replay_output",
    "_restore_failed_replay_output",
    "_delete_successful_replay_backup",
    "_delete_path_best_effort",
    "_delete_failed_streaming_outputs",
    "_run_post_recording_followup",
    "start_this_play",
    "judge_play_and_record",
    "reset_work_pram",
    "update_player_btn_is_paused",
    "_start_streaming_recording",
    "_start_blocking_recording",
    "on_audio_chunk_received_playrec",
    "on_audio_chunk_received_rec",
    "on_clicked_replayer_btn",
    "_project_workflow_state",
    "eventFilter",
}
TARGET_MODULE_HELPERS = set()


def _recording_delay_keys(value):
    keys = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).startswith("recording_start_delay"):
                keys.append(str(key))
            keys.extend(_recording_delay_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_recording_delay_keys(child))
    return keys


def _recording_delay_frame_keys(value):
    return [key for key in _recording_delay_keys(value) if key == "recording_start_delay_frames"]


def _assert_no_recording_delay_frame_keys(value):
    assert _recording_delay_frame_keys(value) == []


@pytest.fixture(scope="module")
def qapp():
    import os
    from PyQt5.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def _option_list_logger():
    return types.SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)


def _namespace_sample_rate_from_device(device):
    if isinstance(device, dict) and device.get("samplerate") not in (None, ""):
        return int(device["samplerate"])
    return 48000


def _namespace_resolve_input_sample_rate(mic):
    return types.SimpleNamespace(ok=True, sample_rate=_namespace_sample_rate_from_device(mic), message="")


def _namespace_resolve_duplex_sample_rate(mic, speaker):
    input_rate = _namespace_sample_rate_from_device(mic)
    output_rate = _namespace_sample_rate_from_device(speaker)
    if input_rate != output_rate:
        return types.SimpleNamespace(ok=False, sample_rate=None, message="sample rate mismatch")
    return types.SimpleNamespace(ok=True, sample_rate=input_rate, message="")


def _build_method_namespace():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sequence_window_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )

    module_helper_sources = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_MODULE_HELPERS:
            module_helper_sources[node.name] = textwrap.dedent(ast.get_source_segment(source, node))

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
        "QIcon": lambda *args, **kwargs: ("icon", args, kwargs),
        "QSize": lambda *args, **kwargs: ("size", args, kwargs),
        "pyqtSlot": lambda *args, **kwargs: (lambda func: func),
        "MessageBox": DummyMessageBox,
        "LoadUiConfig": DummyLoadUiConfig,
        "QEvent": DummyEventType,
        "Qt": DummyQt,
        "QLineEdit": DummyLineEdit,
        "datetime": datetime,
        "json": json,
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
        "save_audio_with_calibration_metadata": lambda save_path, audio, sr=44100, calibration_metadata=None, logger=None, bit_depth=32: namespace[
            "save_audio_simple"
        ](save_path, audio, sr, bit_depth=bit_depth),
        "append_wav_calibration_metadata": lambda *args, **kwargs: True,
        "build_recording_wav_calibration_metadata": lambda input_channels, hardware_id=None, logger=None: {
            "recorded_channels": [
                {
                    "wav_channel_index": wav_channel_index,
                    "v2pa_factor": None,
                    "standard_spl": None,
                    "calibrated": False,
                }
                for wav_channel_index, _channel in enumerate(input_channels)
            ]
        },
        "save_recorded_data_to_json": lambda *args, **kwargs: None,
        "normalize_float_bit_depth": normalize_float_bit_depth,
        "normalize_play_record_detail": normalize_play_record_detail,
        "normalize_record_only_detail": normalize_record_only_detail,
        "resolve_input_sample_rate": _namespace_resolve_input_sample_rate,
        "resolve_duplex_sample_rate": _namespace_resolve_duplex_sample_rate,
        "np": np,
        "SoundcardAudioProcessor": None,
        "re": re,
        "Mapping": Mapping,
        "MappingProxyType": MappingProxyType,
        "OrderedDict": OrderedDict,
        "ReplayRequested": ReplayRequested,
        "legacy_recording_session_snapshot": legacy_recording_session_snapshot,
        "uuid4": uuid4,
        "time": types.SimpleNamespace(monotonic=lambda: 1000.0),
        "os": types.SimpleNamespace(
            path=types.SimpleNamespace(
                exists=lambda path: False,
                basename=__import__("os").path.basename,
                splitext=__import__("os").path.splitext,
            ),
            remove=lambda path: None,
            replace=lambda src, dst: None,
        ),
    }

    for helper_name in TARGET_MODULE_HELPERS:
        exec(module_helper_sources[helper_name], namespace)

    for method_name, method_source in method_sources.items():
        exec(method_source, namespace)

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
        self._enabled_state = True
        self.disabled_values = []
        self.icon_values = []
        self.icon_size_values = []

    def setDisabled(self, value):
        self.disabled = value
        self._enabled_state = not bool(value)
        self.disabled_values.append(bool(value))

    def setEnabled(self, value):
        self.enabled = value
        self._enabled_state = bool(value)

    def isEnabled(self):
        return self._enabled_state

    def setIcon(self, value):
        self.icon_values.append(value)

    def setIconSize(self, value):
        self.icon_size_values.append(value)


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


def _install_sequence_load_config(namespace):
    namespace["LoadUiConfig"] = types.SimpleNamespace(
        get_rec_and_play_dict_base_sequence_dict=lambda data_struct, total_time=None, recording_start_delay_ms=None: (
            {
                "data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "sr": data_struct.sample_rate,
            },
            {
                "sr": data_struct.sample_rate,
                "num_frames": 3,
            },
        )
    )


def _bind_method(obj, namespace, method_name):
    return namespace[method_name].__get__(obj, type(obj))


def test_set_audio_devices_available_stores_state_and_refreshes_player_button():
    namespace = _build_method_namespace()
    calls = []
    obj = types.SimpleNamespace(
        configuration_controller=types.SimpleNamespace(
            set_audio_devices_available=lambda available, message: calls.append(
                (available, message)
            )
        )
    )

    namespace["set_audio_devices_available"](obj, False, "设备不可用")

    assert calls == [(False, "设备不可用")]


def test_external_workflow_entries_route_through_guarded_start_paths():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    expected_entries = {
        "_tcp_run_test",
        "_commit_barcode",
        "on_sensor_triggered",
        "on_shortcut_triggered",
        "start_this_play",
        "judge_play_and_record",
    }
    assert expected_entries.issubset(methods)

    def calls(method_name, target_name):
        for node in ast.walk(methods[method_name]):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == target_name:
                    return True
                if isinstance(func, ast.Name) and func.id == target_name:
                    return True
        return False

    assert calls("_tcp_run_test", "handle_tcp_run_test")
    assert calls("_commit_barcode", "commit_barcode")
    assert calls("on_sensor_triggered", "handle_optical_trigger")
    assert calls("on_shortcut_triggered", "handle_shortcut_trigger")
    assert calls("start_this_play", "request_start")
    assert calls("judge_play_and_record", "request_start")

    source_text = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_active_instance_ref" not in source_text








def test_migrated_recording_behaviors_are_owned_by_formal_mvc_ports():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter
    from ui.sequence.sequence_recording_worker import (
        SequenceStreamingRecordingService,
        StreamingRecordingWorker,
    )

    assert callable(BlockingRecordingAdapter.prepare)
    assert callable(BlockingRecordingAdapter.acquire)
    assert callable(BlockingRecordingAdapter.transaction)
    assert callable(SequenceStreamingRecordingService.start)
    assert callable(StreamingRecordingWorker.run)


@pytest.mark.parametrize("rejection", ["busy", "configuration"])
def test_replay_button_rejection_never_calls_legacy_recorder(qapp, rejection):
    namespace = _build_method_namespace()

    class FakeWindow:
        pass

    window = FakeWindow()
    window.sequence_event_bus = SequenceEventBus()
    window.workflow_model = SequenceWorkflowModel()
    window.workflow_model.configuration_generation = 4
    inputs = RecordingAdmissionInputs(
        configuration_generation=4,
        product_model="MODEL",
        serial_number="SN",
        scanner_enabled=False,
        current_recorded_count=4,
        last_play_count=4,
        recorded_path="D:/recordings/record-4.wav",
        recorded_signal_info={"file_path": "record-4.wav"},
        stimulus_data=None,
        stimulus_info=None,
    )
    window.recording_admission_service = RecordingAdmissionService(
        raw_inputs=lambda: inputs
    )
    window.on_clicked_replayer_btn = _bind_method(
        window, namespace, "on_clicked_replayer_btn"
    )
    starts = []
    controller = SequenceWorkflowController(
        window.workflow_model,
        window.sequence_event_bus,
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config=(), analysis_config={}
        ),
        replay_readiness=window.recording_admission_service.replay_readiness,
        session_snapshot_factory=window.recording_admission_service.session_snapshot,
    )
    bridge = LegacyRecordingAdmissionBridge(
        window.sequence_event_bus,
        lambda admission, terminal: starts.append((admission, terminal)) or True,
        workflow_generation_provider=lambda: window.workflow_model.workflow_generation,
    )
    window.sequence_event_bus.events.workflow_command_rejected.connect(
        window.recording_admission_service.discard_rejected
    )
    if rejection == "busy":
        window.workflow_model.phase = WorkflowPhase.RECORDING

    assert window.on_clicked_replayer_btn() is True
    if rejection == "configuration":
        inputs.configuration_generation = 5
    for _ in range(5):
        qapp.processEvents()

    assert starts == []
    assert window.recording_admission_service.pending_replay_count == 0

    bridge.disconnect()
    controller.disconnect()


def test_replay_button_connection_has_no_direct_legacy_replay_call():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "self.replayer_btn.clicked.connect(self.on_clicked_replayer_btn)" in source
    assert "self.replayer_btn.clicked.connect(lambda: self.judge_play_and_record(is_replay=True))" not in source


def test_unadmitted_replay_button_targets_are_bounded_and_immutable():
    namespace = _build_method_namespace()
    inputs = RecordingAdmissionInputs(
        configuration_generation=2,
        product_model="MODEL",
        serial_number="SN",
        scanner_enabled=False,
        current_recorded_count=2,
        last_play_count=2,
        recorded_path="D:/recordings/record-2.wav",
        recorded_signal_info={"file_path": "record-2.wav"},
        stimulus_data=None,
        stimulus_info=None,
    )
    service = RecordingAdmissionService(raw_inputs=lambda: inputs)
    window = types.SimpleNamespace(
        sequence_event_bus=SequenceEventBus(),
        recording_admission_service=service,
    )
    window.on_clicked_replayer_btn = _bind_method(
        window, namespace, "on_clicked_replayer_btn"
    )

    for _ in range(10_000):
        assert window.on_clicked_replayer_btn() is True

    assert service.pending_replay_count == 64
    assert all(
        type(target) is MappingProxyType
        for target in service._pending_replays.values()
    )






def test_update_player_button_requires_audio_devices_available():
    namespace = _build_method_namespace()
    obj = types.SimpleNamespace(
        sequence_config=[{"seq1": {}}],
        audio_devices_available=False,
        player_status_flag=False,
        _record_workflow_busy=False,
        player_btn=FakeButton(),
    )

    namespace["update_player_btn_is_paused"](obj)

    assert obj.player_btn.disabled_values[-1] is True


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
    window.current_recorded_count = 1
    window.last_play_count = 1
    window._pending_recorded_count = None
    window._active_recording_is_replay = False
    window._active_recording_output_path = None
    window._active_replay_output_temp_path = None
    window._active_replay_output_backup_path = None
    window._active_replay_output_replaced = False
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
    window.checked_work_status_message = lambda: False
    window.reset_work_pram = lambda label, count=None: reset_result
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

    class TriggerViewAdapter:
        @property
        def serial_input(self):
            return window.lineedit_s_or_n

        @property
        def product_input(self):
            return window.lineedit_type

        @property
        def count_input(self):
            return window.lineedit_count

        def is_scanner_checked(self):
            return window.barcode_scanner_box.isChecked()

        def is_serial_enabled(self):
            return True

        def serial_text(self):
            return window.lineedit_s_or_n.text()

        def set_serial_text(self, text):
            window.lineedit_s_or_n.setText(text)

        def clear_serial_text(self):
            window.lineedit_s_or_n.clear()

        def focus_widget(self):
            return namespace["QApplication"].focusWidget()

        def focus_serial_input(self, *, select_all=False):
            window.lineedit_s_or_n.setFocus()
            if select_all:
                window.lineedit_s_or_n.selectAll()

        def prepare_for_continuous_scan(self):
            window._close_analysis_windows()

        def show_invalid_barcode(self, barcode, invalid_chars):
            namespace["MessageBox"].warning(window, "条形码包含特殊字符", f"{barcode}: {invalid_chars}")

        def show_regex_rejection(self, rule, sn_text, value_label, retry_hint):
            namespace["MessageBox"].warning(
                window,
                "SN 正则校验失败",
                f"当前 SN 内容不符合已启用规则：\n\n"
                f"规则名称：{rule['name']}\n"
                f"规则表达式：{rule['pattern']}\n"
                f"{value_label}：{sn_text or '（空）'}\n\n"
                f"{retry_hint}",
            )

        def show_mode_rejection(self, trigger_source, mode):
            window._show_external_trigger_mode_warning(trigger_source, mode)

        def show_busy_rejection(self, _trigger_source):
            return None

        def show_workflow_rejection(self, _reason):
            return None

        def is_protected_input_widget(self, _widget):
            return False

    window.trigger_model = SequenceTriggerModel()
    window.trigger_view = TriggerViewAdapter()
    window.trigger_commands = []
    window.trigger_controller = SequenceTriggerController(
        window.trigger_model,
        window.trigger_view,
        start_publisher=window.trigger_commands.append,
        configuration_generation_provider=lambda: 0,
        workflow_active_provider=lambda: bool(
            window._record_workflow_busy or window.player_status_flag
        ),
        external_mode_available_provider=lambda: window.mode in {"RECORD_ONLY", "PLAY_AND_RECORD"},
        acquisition_mode_provider=lambda: window.mode,
        regex_rule_loader=lambda: window._load_selected_sn_regex_rule(),
        monotonic=namespace["time"].monotonic,
        command_id_factory=(lambda counter=iter(range(1000)): f"test-trigger-{next(counter)}"),
        debounce_timer=window._barcode_debounce_timer,
        logger=window.default_logger,
    )
    window._barcode_router = BarcodeRouter(window.trigger_controller)
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
    window._real_reset_work_pram = _bind_method(window, namespace, "reset_work_pram")
    window._start_streaming_recording = _bind_method(window, namespace, "_start_streaming_recording")
    window._start_blocking_recording = _bind_method(window, namespace, "_start_blocking_recording")
    window.on_audio_chunk_received_playrec = _bind_method(window, namespace, "on_audio_chunk_received_playrec")
    window.on_audio_chunk_received_rec = _bind_method(window, namespace, "on_audio_chunk_received_rec")
    window.eventFilter = _bind_method(window, namespace, "eventFilter")
    return window








def test_option_list_new_record_only_item_uses_default_recording_start_delay(
    qapp, tmp_path, monkeypatch
):
    from ui import operation_sequence

    empty_config = tmp_path / "empty.json"
    empty_config.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        operation_sequence,
        "load_acquisition_defaults",
        lambda logger=None: {
            "RECORD_ONLY": {
                "total_time": 2.0,
                "sample_rate": 48000,
                "monitor_playback": False,
                "monitor_input_channel": 0,
                "monitor_gain_db": 0.0,
                "use_streaming_recording": False,
                "recording_start_delay_ms": 220.0,
            },
            "PLAY_AND_RECORD": {
                "use_streaming_recording": False,
                "recording_start_delay_ms": 100.0,
            },
        },
    )
    option_list = operation_sequence.OptionList(
        logger=_option_list_logger(),
        using_config_path=str(empty_config),
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
        mic_channels=[0],
    )

    option_list.set_sound_item("录制音频")

    assert option_list.config[0].mode == "RECORD_ONLY"
    assert option_list.config[0].detail["recording_start_delay_ms"] == 220.0
    assert "recording_start_delay_frames" not in option_list.config[0].detail


def test_option_list_new_play_record_item_uses_default_recording_start_delay(
    qapp, tmp_path, monkeypatch
):
    from ui import operation_sequence

    empty_config = tmp_path / "empty.json"
    empty_config.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        operation_sequence,
        "load_acquisition_defaults",
        lambda logger=None: {
            "PLAY_AND_RECORD": {
                "use_streaming_recording": True,
                "recording_start_delay_ms": 320.0,
            },
            "RECORD_ONLY": {"recording_start_delay_ms": 100.0},
        },
    )

    option_list = operation_sequence.OptionList(
        logger=_option_list_logger(),
        using_config_path=str(empty_config),
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
        mic_channels=[0],
    )
    option_list.load_stimulus_config = lambda: (
        True,
        {
            "stimulus_info": {"total_time": 1.0, "sample_rate": 48000},
            "total_time": 1.0,
            "sample_rate": 48000,
        },
    )

    option_list.set_sound_item("播放与录制")

    assert option_list.config[0].mode == "PLAY_AND_RECORD"
    assert option_list.config[0].detail["recording_start_delay_ms"] == 320.0
    assert option_list.config[0].detail["use_streaming_recording"] is True
    assert "recording_start_delay_frames" not in option_list.config[0].detail


def test_option_list_loads_old_sequence_config_with_default_recording_start_delay(
    qapp, tmp_path
):
    from ui import operation_sequence

    config_path = tmp_path / "old_sequence.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "seq1": {
                        "acq": {
                            "name": "录制音频",
                            "mode": "RECORD_ONLY",
                            "detail": {"total_time": 1.0, "sample_rate": 48000},
                        },
                        "analysis_list": {
                            "display_sequence": [],
                            "auto_analysis": False,
                        },
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    option_list = operation_sequence.OptionList(
        logger=_option_list_logger(),
        using_config_path=str(config_path),
        mic={"name": "mic"},
        speaker={"name": "speaker", "max_output_channels": 2},
        mic_channels=[0],
    )

    assert option_list.config[0].detail["recording_start_delay_ms"] == 100.0
    assert "recording_start_delay_frames" not in option_list.config[0].detail

    saved_config = [option_list.config[0].config_info]
    assert saved_config[0]["seq1"]["acq"]["detail"]["recording_start_delay_ms"] == 100.0
    _assert_no_recording_delay_frame_keys(saved_config)






























































































def test_sn_lock_restores_previously_disabled_scanner_checkbox_state():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.barcode_scanner_box = FakeCheckBox(enabled=False)

    window._set_sn_input_recording_read_only(True)
    assert window.barcode_scanner_box.isEnabled() is False

    window._set_sn_input_recording_read_only(False)
    assert window.lineedit_s_or_n.isReadOnly() is False
    assert window.barcode_scanner_box.isEnabled() is False








def test_sequence_config_update_keeps_reloaded_play_record_sample_rate():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        _ensure_configuration_projection_hooks=lambda window, controller: None
    )
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
    def _reload_stimulus_config():
        window.data_struct.sample_rate = window.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"][
            "sample_rate"
        ]

    window.configuration_controller = types.SimpleNamespace(
        on_sequence_config_updated=_reload_stimulus_config
    )

    assert window.on_sequence_config_updated() is None

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
    assert window.trigger_commands == []
    assert window.trigger_model.last_committed_barcode is None
    assert window.trigger_model.last_committed_barcode_time == 0.0
    assert window._barcode_debounce_timer.isActive() is False
    assert window.trigger_model.barcode_first_char_ts is None
    assert window.trigger_model.barcode_last_char_ts is None
    assert window.trigger_model.barcode_capture_buffer == ""


def test_commit_barcode_when_idle_still_updates_sn_and_starts_test():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}

    window._commit_barcode("SN-123", source="hid")

    assert window.lineedit_s_or_n.text() == "SN-123"
    assert window.lineedit_s_or_n.set_text_calls == ["SN-123"]
    assert len(window.trigger_commands) == 1
    assert window.trigger_commands[0].source == "hid"
    assert window.trigger_commands[0].label == "not_labeled"
    assert window.close_analysis_calls == 1
    assert window.trigger_model.last_committed_barcode == "SN-123"
    assert window.trigger_model.last_committed_barcode_time == 1000.0








def test_tcp_run_test_allows_invalid_sn_when_validation_is_explicitly_skipped():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("BAD-SN")
    window.barcode_scanner_box = FakeCheckBox(checked=True)
    window._load_selected_sn_regex_rule = lambda: {"name": "sn-rule", "pattern": r"SN-\d{3}"}

    window._tcp_run_test("not_labeled", skip_sn_regex_validation=True)

    assert len(window.trigger_commands) == 1
    assert window.trigger_commands[0].source == "tcp"
    assert window.trigger_commands[0].label == "not_labeled"
    assert window.trigger_commands[0].skip_sn_regex_validation is True
    assert namespace["MessageBox"].warnings == []




def test_tcp_analysis_result_payload_uses_summary_and_recorded_file_name():
    payload = SequenceAnalysisController.build_tcp_result_payload(
        {"recorded_path": r"D:\audio\OH2P-001.wav"},
        {"SPL": (True, "ok"), "FR": (True, "ok")},
        datetime.now(),
    )

    assert set(payload) == {"TimeStamp", "Label", "FileName"}
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}$", payload["TimeStamp"])
    assert payload["Label"] == "OK"
    assert payload["FileName"] == "OH2P-001.wav"


def test_tcp_analysis_result_payload_defaults_to_not_labeled_without_summary():
    payload = SequenceAnalysisController.build_tcp_result_payload(
        {
            "recorded_path": "",
            "recorded_signal_info": {
                "file_path": "stored_data/OH2P-002.wav"
            },
        },
        {},
        datetime.now(),
    )

    assert payload["Label"] == "not_labeled"
    assert payload["FileName"] == "OH2P-002.wav"


def test_tcp_analysis_result_callback_sends_json_to_current_tcp_client():
    sends = []
    fake_tcp_server = types.SimpleNamespace(
        send_to_current_client=lambda message: sends.append(message) or True
    )
    owner = SequenceAnalysisTransportController(
        bus=SequenceEventBus(),
        authorization_provider=lambda _event: False,
        authorization_consumer=lambda _event: False,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: fake_tcp_server,
    )
    payload = SequenceAnalysisController.build_tcp_result_payload(
        {"recorded_path": "OH2P-003.wav"},
        {"SPL": (False, "ng")},
        datetime.now(),
    )
    assert owner.send_payload(payload)

    assert len(sends) == 1
    sent_payload = json.loads(sends[0])
    assert sent_payload["Label"] == "NG"
    assert sent_payload["FileName"] == "OH2P-003.wav"


def test_tcp_analysis_result_callback_skips_without_tcp_server():
    messages = []
    owner = SequenceAnalysisTransportController(
        bus=SequenceEventBus(),
        authorization_provider=lambda _event: False,
        authorization_consumer=lambda _event: False,
        tcp_enabled_provider=lambda: True,
        tcp_server_provider=lambda: None,
        logger=types.SimpleNamespace(
            warning=lambda message: messages.append(message)
        ),
    )

    assert owner.send_payload(
        {"TimeStamp": "t", "Label": "OK", "FileName": "a.wav"}
    ) is False
    assert any("tcp_callback_skip" in message for message in messages)


def test_tcp_run_test_allows_play_and_record_mode():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="PLAY_AND_RECORD")
    window._tcp_run_test("OK", skip_sn_regex_validation=True)

    assert len(window.trigger_commands) == 1
    assert window.trigger_commands[0].source == "tcp"
    assert window.trigger_commands[0].label == "OK"
    assert window.trigger_commands[0].skip_sn_regex_validation is True
    assert namespace["MessageBox"].warnings == []


def test_tcp_run_test_blocks_unsupported_mode_with_friendly_message():
    namespace = _build_method_namespace()
    namespace["SequenceWindow"] = types.SimpleNamespace(
        tcp_server=types.SimpleNamespace(client_address=("127.0.0.1", 5000))
    )
    window = _build_fake_window(namespace, use_streaming=True, mode="IMPORT_AUDIO")

    window._tcp_run_test("OK", skip_sn_regex_validation=True)

    assert window.trigger_commands == []
    warnings = namespace["MessageBox"].warnings
    assert len(warnings) == 1
    assert "IMPORT_AUDIO" in warnings[0][0][2]
    assert "TCP" in warnings[0][0][2]




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
    window.trigger_model.barcode_first_char_ts = 11.0
    window.trigger_model.barcode_last_char_ts = 11.2
    window._barcode_debounce_timer.active = True

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    window.trigger_controller.handle_keypress(
        window.lineedit_s_or_n,
        FakeKeyEvent(QtEvent.KeyPress, QtCore.Key_Backspace),
    )

    assert window.trigger_model.sn_textchange_manual_guard is True
    assert window.trigger_model.barcode_first_char_ts is None
    assert window.trigger_model.barcode_last_char_ts is None
    assert window._barcode_debounce_timer.isActive() is False


def test_sn_full_selection_rearms_textchange_auto_commit():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")
    window.lineedit_s_or_n.setSelectedText("SN-OLD")
    window.trigger_model.sn_textchange_manual_guard = True

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    window.trigger_controller.handle_keypress(
        window.lineedit_s_or_n,
        FakeKeyEvent(QtEvent.KeyPress, ord("S"), text="S"),
    )

    assert window.trigger_model.sn_textchange_manual_guard is False


def test_sn_ctrl_z_is_swallowed_without_breaking_startup_logic():
    namespace = _build_method_namespace()
    window = _build_fake_window(namespace, use_streaming=True, mode="RECORD_ONLY")
    window.lineedit_s_or_n = FakeLineEdit("SN-OLD")

    namespace["QApplication"].focusWidget = staticmethod(lambda: window.lineedit_s_or_n)
    handled = window.trigger_controller.handle_keypress(
        window.lineedit_s_or_n,
        FakeKeyEvent(
            QtEvent.KeyPress,
            QtCore.Key_Z,
            text="z",
            modifiers=QtCore.ControlModifier,
        ),
    )

    assert handled is True
