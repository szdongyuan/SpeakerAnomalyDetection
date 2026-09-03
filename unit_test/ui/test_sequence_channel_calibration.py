import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication

from consts import error_code
from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
)
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_serial_trigger_ops import (
    SequenceWidgetSerialTriggerOpsMixin,
)
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.analysis_channel_preflight import (
    MULTI_CHANNEL_ANALYSIS_TYPES,
    REQUIRED_CHANNEL_ANALYSIS_TYPES,
    preflight_analysis_channels,
)
from ui.ui_analysis_config.config_normalization import (
    normalize_analysis_channel,
    normalize_analysis_channels,
)
from base.wav_calibration_metadata import resolve_wav_channel_v2pa_factor


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_OPS = ROOT / "ui" / "sequence" / "sequence_widget_analysis_ops.py"
DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 4,
}


class RecordingSnapshotHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self, *, streaming=False):
        self.events = []
        self._streaming = streaming
        self._record_workflow_busy = False
        self.last_play_count = 10
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.player_status_flag = True
        self.clicked_player_flag = True
        self.streaming_buffer_multi = []
        self.replayer_btn = SimpleNamespace(setDisabled=lambda _value: None)
        self.data_btn = SimpleNamespace(setDisabled=lambda _value: None)
        self.mic = DEVICE
        self._active_input_channels = [9]
        self.recorded_path = "recorded.wav"
        self.recorded_signal_info = {}
        self.default_logger = SimpleNamespace(
            error=mock.Mock(), info=mock.Mock(), warning=mock.Mock()
        )
        self.streaming_poll_timer = SimpleNamespace(start=lambda _ms: None)
        self._unlock_sn_after_recording_if_needed = mock.Mock()
        self._drain_queued_directional_trigger = mock.Mock()
        self._on_serial_product_runtime_error = mock.Mock(return_value=False)
        self.paused_updates = 0

    def checked_work_status_message(self):
        return False

    def _clear_plot_area(self):
        return None

    def _cleanup_streaming_resources(self):
        return None

    def update_player_btn_is_playing(self):
        return None

    def update_player_btn_is_paused(self):
        self.paused_updates += 1

    def reset_work_pram(self, _label, count=None):
        self._active_input_channels = [0, 2]
        self.events.append(("reset", self._active_input_channels))
        return {"input_channels": [0, 2]}, 48000

    def _should_use_streaming_recording(self):
        return self._streaming

    def _begin_recent_session_for_current_run(self):
        return None

    def _start_process_recording(self, _recorded_dict, _sample_rate, *, tcp_completion_address=None):
        self.events.append(
            ("process", self._recording_wav_calibration_metadata)
        )


def test_recording_attempt_captures_once_after_reset_before_process_start(monkeypatch):
    host = RecordingSnapshotHost()
    snapshots = []

    def build(channels, device):
        snapshot = {"attempt": len(snapshots) + 1}
        snapshots.append(snapshot)
        host.events.append(("snapshot", list(channels), device, snapshot))
        return snapshot

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.build_recording_wav_calibration_metadata",
        build,
    )

    host.judge_play_and_record()

    assert [event[0] for event in host.events] == ["reset", "snapshot", "process"]
    assert host.events[1][1] == [0, 2]
    assert host.events[2][1] is snapshots[0]
    assert host._recording_wav_calibration_metadata is snapshots[0]


def test_replay_replaces_snapshot_and_each_attempt_reads_once(monkeypatch):
    host = RecordingSnapshotHost()
    built = []

    def build(_channels, _device):
        snapshot = {"attempt": len(built) + 1}
        built.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.build_recording_wav_calibration_metadata",
        build,
    )

    host.judge_play_and_record()
    first = host._recording_wav_calibration_metadata
    host._record_workflow_busy = False
    host.judge_play_and_record(is_replay=True)

    assert len(built) == 2
    assert first is built[0]
    assert host._recording_wav_calibration_metadata is built[1]


def test_streaming_process_start_keeps_calibration_snapshot(monkeypatch):
    host = RecordingSnapshotHost(streaming=True)
    snapshot = {"recorded_channels": [{"wav_channel_index": 0}]}
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.build_recording_wav_calibration_metadata",
        lambda channels, device: snapshot,
    )
    host.judge_play_and_record()
    assert host.events[-1] == ("process", snapshot)
    assert host._recording_wav_calibration_metadata is snapshot


@pytest.mark.parametrize(
    "error",
    [MicCalibrationFormatError("bad"), MicCalibrationIOError("denied")],
)
def test_snapshot_file_error_logs_stores_none_and_still_records(monkeypatch, error):
    host = RecordingSnapshotHost()
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.build_recording_wav_calibration_metadata",
        mock.Mock(side_effect=error),
    )

    host.judge_play_and_record()

    assert host._recording_wav_calibration_metadata is None
    assert host.events[-1][0] == "process"
    assert str(error) in host.default_logger.error.call_args.args[0]


def test_snapshot_does_not_swallow_programming_error(monkeypatch):
    host = RecordingSnapshotHost()
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.build_recording_wav_calibration_metadata",
        mock.Mock(side_effect=RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        host.judge_play_and_record()

    assert not any(event[0] == "process" for event in host.events)
    assert host._recording_wav_calibration_metadata is None
    assert host._record_workflow_busy is False
    assert host.player_status_flag is False
    host._unlock_sn_after_recording_if_needed.assert_called_once_with()
    assert host.paused_updates == 1
    host._drain_queued_directional_trigger.assert_called_once_with()
    host._on_serial_product_runtime_error.assert_called_once()
    assert "初始化录音失败" in host._on_serial_product_runtime_error.call_args.args[0]
    assert "bug" in host.default_logger.error.call_args.args[0]


def _load_methods(method_names, extra_globals):
    module_tree = ast.parse(ANALYSIS_OPS.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceWidgetAnalysisOpsMixin"
    )
    methods = [
        next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        for method_name in method_names
    ]
    test_class = ast.ClassDef(
        name="TestSequence",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = dict(extra_globals)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[test_class], type_ignores=[])
            ),
            str(ANALYSIS_OPS),
            "exec",
        ),
        namespace,
    )
    return namespace["TestSequence"]


class FakeAnalysis:
    def __init__(self, key, events):
        self.key = key
        self.events = events

    def calculate_spl(self):
        self.events.append(("calculate", self.key))
        judgment = getattr(self, "analysis_config", {}).get("_test_judgment")
        if isinstance(judgment, bool):
            self.data_struct.analysis_result_dict[self.key] = (
                judgment,
                0.0,
            )
        return True


class FakeSpec:
    def __init__(self, key, events):
        self.key = key
        self.events = events

    def calculate_spec(self):
        self.events.append(("calculate_spec", self.key))


class FakeFBA:
    def __init__(self, key, events):
        self.key = key
        self.events = events

    def calculate_fba(self):
        self.events.append(("calculate_fba", self.key))
        return True


class FakeRSC:
    def __init__(self, key, events):
        self.key = key
        self.events = events

    def calculate_reference_spectrum(self):
        self.events.append(("calculate_rsc", self.key))
        return True


def build_sequence(
    *,
    factor_map=None,
    loader_error=None,
    active_input_channels=None,
    import_audio=False,
    items=None,
    legacy_factor=9.5,
    wav_metadata=None,
    wav_resolver=resolve_wav_channel_v2pa_factor,
):
    events = []
    created = []

    def make_analysis(key):
        instance = FakeAnalysis(key, events)
        created.append(instance)
        return instance

    def make_rsc(key):
        instance = FakeRSC(key, events)
        created.append(instance)
        return instance

    def make_spec(key):
        instance = FakeSpec(key, events)
        created.append(instance)
        return instance

    def make_fba(key):
        instance = FakeFBA(key, events)
        created.append(instance)
        return instance

    def get_class_mapping():
        return {
            "AI": make_analysis,
            "FFT": make_analysis,
            "LOUD": make_analysis,
            "SPL": make_analysis,
            "Spec": make_spec,
            "FBA": make_fba,
            "LP": make_analysis,
            "RSC": make_rsc,
        }

    loader = mock.Mock(
        side_effect=loader_error,
        return_value={} if factor_map is None else factor_map,
    )
    messages = SimpleNamespace(
        warning=mock.Mock(
            side_effect=lambda _parent, title, body: events.append(
                ("warning", title, body)
            )
        ),
        critical=mock.Mock(
            side_effect=lambda _parent, title, body: events.append(
                ("critical", title, body)
            )
        ),
    )
    cls = _load_methods(
        [
            "_reset_live_mic_calibration_batch",
            "_live_batch_requires_mic_calibration",
            "_prepare_live_mic_calibration_batch",
            "_resolve_live_mic_channel_v2pa_factor",
            "_show_missing_mic_channel_calibration_warning",
            "_abort_live_mic_calibration_batch",
            "_show_analysis_channel_preflight_warning",
            "_prepare_imported_wav_calibration_batch",
            "_show_imported_wav_calibration_warning",
            "_run_analysis_impl",
            "instance_analysis_class",
        ],
        {
            "load_mic_channel_v2pa_factors": loader,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": messages,
            "QSize": QSize,
            "preflight_analysis_channels": preflight_analysis_channels,
            "REQUIRED_CHANNEL_ANALYSIS_TYPES": REQUIRED_CHANNEL_ANALYSIS_TYPES,
            "MULTI_CHANNEL_ANALYSIS_TYPES": MULTI_CHANNEL_ANALYSIS_TYPES,
            "normalize_analysis_channel": normalize_analysis_channel,
            "normalize_analysis_channels": normalize_analysis_channels,
            "resolve_wav_channel_v2pa_factor": wav_resolver,
            "extract_ai_runtime_state": lambda *_args: {
                "has_ai_analysis": False,
                "scores": {},
            },
        },
    )
    sequence = cls()
    sequence._get_legacy_analysis_class_mapping = get_class_mapping
    sequence.mic = DEVICE
    sequence.v2pa_factor = legacy_factor
    sequence._active_input_channels = list(active_input_channels or [0])
    sequence._is_import_audio_mode = lambda: import_audio
    sequence.data_struct = SimpleNamespace(
        analysis_result_dict={},
        store_wave_data_multi=None,
        wav_calibration_warning_shown=False,
        wav_calibration_metadata=wav_metadata,
    )
    sequence.analysis_window = []
    sequence._analysis_result_summary_window = None
    sequence.default_logger = SimpleNamespace(
        error=mock.Mock(), info=mock.Mock(), warning=mock.Mock()
    )
    sequence._close_analysis_windows = lambda: sequence.analysis_window.clear()
    sequence.screen = lambda: SimpleNamespace(
        size=lambda: SimpleNamespace(width=lambda: 1600, height=lambda: 900)
    )
    sequence._hide_analysis_window = mock.Mock()
    sequence._capture_excel_export_cache = mock.Mock()
    sequence._maybe_export_excel_results = mock.Mock()
    sequence._can_output_ok_ng = mock.Mock(return_value=(False, ""))
    sequence._sync_left_panel_analysis_details = mock.Mock()
    sequence.count_board = SimpleNamespace(mode="view")
    sequence.recorded_signal_info = {}
    sequence.analysis_config = {
        "display_sequence": [key for key, _item_type, _params in (items or [])],
        **{
            key: {"type": item_type, **params}
            for key, item_type, params in (items or [])
        },
    }
    return sequence, loader, messages, events, created


def test_live_batch_uses_physical_channel_for_factor_and_local_column_for_data():
    sequence, _loader, _messages, _events, _created = build_sequence(
        active_input_channels=[0, 2],
    )
    sequence._live_mic_channel_v2pa_factors = {0: 1.25, 2: 3.5}
    sequence._missing_mic_calibration_channels = []
    sequence._missing_mic_calibration_channel_set = set()

    sequence.instance_analysis_class(
        "SPL", "SPL", {"analysis_channel": 2}
    )

    instance = sequence.analysis_window[0]
    assert instance.v2pa_factor == 3.5
    assert instance.analysis_config["analysis_channel"] == 1


def test_live_batch_loads_once_and_warns_before_calculation_in_first_seen_order():
    sequence, loader, messages, events, _created = build_sequence(
        factor_map={},
        active_input_channels=[0, 2],
        items=[
            ("first", "SPL", {"analysis_channel": 2}),
            ("second", "SPL", {"analysis_channel": 0}),
            ("duplicate", "SPL", {"analysis_channel": 2}),
        ],
    )

    result = sequence._run_analysis_impl(show_windows=False)

    assert result is True
    loader.assert_called_once_with(DEVICE)
    messages.warning.assert_called_once()
    _parent, title, body = messages.warning.call_args.args
    assert title == "输入通道未校准"
    assert body == "In3、In1 未校准，结果仅供参考"
    assert events[0] == ("warning", title, body)
    assert [event[0] for event in events[1:]] == [
        "calculate",
        "calculate",
        "calculate",
    ]


def test_calibrated_and_missing_channels_receive_exact_batch_factors():
    sequence, loader, messages, _events, created = build_sequence(
        factor_map={0: 1.25, 2: 3.5},
        active_input_channels=[0, 1, 2],
        items=[
            ("calibrated-0", "SPL", {"analysis_channel": 0}),
            ("missing-1", "SPL", {"analysis_channel": 1}),
            ("calibrated-2", "SPL", {"analysis_channel": 2}),
        ],
    )

    result = sequence._run_analysis_impl(show_windows=False)

    assert result is True
    loader.assert_called_once_with(DEVICE)
    assert [instance.v2pa_factor for instance in created] == [1.25, 1.0, 3.5]
    assert messages.warning.call_args.args[2] == "In2 未校准，结果仅供参考"


def test_missing_channel_warning_deduplication_resets_for_next_run():
    sequence, loader, messages, _events, _created = build_sequence(
        factor_map={},
        active_input_channels=[2],
        items=[
            ("first", "SPL", {"analysis_channel": 2}),
            ("duplicate", "SPL", {"analysis_channel": 2}),
        ],
    )

    sequence._run_analysis_impl(show_windows=False)
    sequence._run_analysis_impl(show_windows=False)

    assert loader.call_count == 2
    assert messages.warning.call_count == 2
    assert [call.args[2] for call in messages.warning.call_args_list] == [
        "In3 未校准，结果仅供参考",
        "In3 未校准，结果仅供参考",
    ]


@pytest.mark.parametrize(
    "error",
    [MicCalibrationFormatError("bad"), MicCalibrationIOError("denied")],
)
def test_calibration_file_error_aborts_whole_live_batch(error):
    sequence, loader, messages, events, created = build_sequence(
        loader_error=error,
        items=[("spl", "SPL", {"analysis_channel": 0})],
    )
    sequence.data_struct.analysis_result_dict["stale"] = (True, 0.0)
    sequence.analysis_window.append(SimpleNamespace(stale=True))

    result = sequence._run_analysis_impl(show_windows=False)

    assert result is False
    loader.assert_called_once_with(DEVICE)
    assert sequence.analysis_window == []
    assert sequence.data_struct.analysis_result_dict == {}
    assert created == []
    assert not any(event[0] == "calculate" for event in events)
    messages.warning.assert_not_called()
    messages.critical.assert_called_once()
    assert messages.critical.call_args.args[1:] == (
        "输入校准文件错误",
        "输入校准文件错误，本次分析已停止",
    )
    assert str(error) in sequence.default_logger.error.call_args.args[0]
    sequence._capture_excel_export_cache.assert_not_called()


@pytest.mark.parametrize("recorded_channels", [None, [0, 7]])
def test_imported_items_use_requested_wav_factor_and_preflight_local_column(recorded_channels):
    resolver = mock.Mock(side_effect=resolve_wav_channel_v2pa_factor)
    sequence, loader, messages, _events, created = build_sequence(
        loader_error=MicCalibrationFormatError("must not load"),
        import_audio=True,
        legacy_factor=99.0,
        wav_resolver=resolver,
        wav_metadata={
            "recorded_channels": [
                {
                    "wav_channel_index": 0,
                    "v2pa_factor": 2.5,
                    "standard_spl": 94.0,
                    "calibrated": True,
                },
                {
                    "wav_channel_index": 1,
                    "v2pa_factor": 7.0,
                    "standard_spl": 114.0,
                    "calibrated": True,
                },
            ]
        },
        items=[
            ("spl", "SPL", {"analysis_channel": 0}),
            ("spec", "Spec", {"analysis_channel": 1}),
            ("fba", "FBA", {"analysis_channel": 0}),
        ],
    )
    sequence.data_struct.store_wave_data_multi = np.zeros((8, 2))
    if recorded_channels is not None:
        for key in sequence.analysis_config["display_sequence"]:
            sequence.analysis_config[key]["analysis_channels"] = recorded_channels

    sequence._run_analysis_impl(show_windows=False)

    loader.assert_not_called()
    messages.warning.assert_not_called()
    messages.critical.assert_not_called()
    assert [instance.v2pa_factor for instance in created] == [2.5, 7.0, 2.5]
    assert [
        instance.analysis_config["analysis_channel"] for instance in created
    ] == [0, 1, 0]
    assert [call.args[1] for call in resolver.call_args_list] == [0, 1, 0]
    assert all(not instance._sequence_multi_channel_expansion for instance in created)


@pytest.mark.parametrize("item_type", ["SPL", "Spec", "FBA", "AI", "LP", "FFT", "LOUD"])
def test_recorded_item_expands_with_physical_calibration_and_local_columns(item_type):
    params = {"analysis_channel": 0, "analysis_channels": [7, 2], "limit_checked": True}
    sequence, loader, messages, _events, created = build_sequence(
        factor_map={2: 2.5, 7: 8.0},
        active_input_channels=[7, 2],
        items=[("item", item_type, params)],
    )

    assert sequence._run_analysis_impl(show_windows=False) is True

    loader.assert_called_once_with(DEVICE)
    messages.warning.assert_not_called()
    assert [instance._sequence_runtime_key for instance in created] == [
        "item--通道3", "item--通道8"
    ]
    assert [instance.analysis_config["analysis_channel"] for instance in created] == [1, 0]
    assert [instance.v2pa_factor for instance in created] == [2.5, 8.0]
    assert [instance._analysis_raw_channel for instance in created] == [2, 7]
    assert all(instance._sequence_analysis_key == "item" for instance in created)
    assert all(instance._sequence_window_key == instance.key for instance in created)
    assert all(instance._sequence_multi_channel_expansion for instance in created)
    assert sequence.analysis_config["display_sequence"] == ["item"]
    assert sequence.analysis_config["item"]["analysis_channels"] == [7, 2]
    assert sequence.analysis_config["item"]["analysis_channel"] == 0


def test_recorded_multichannel_results_and_export_cache_do_not_overwrite():
    sequence, _loader, _messages, _events, _created = build_sequence(
        factor_map={0: 1.0, 2: 3.0},
        active_input_channels=[0, 2],
        items=[("item", "SPL", {
            "analysis_channels": [0, 2],
            "limit_checked": True,
            "_test_judgment": True,
        })],
    )

    sequence._run_analysis_impl(show_windows=False)
    sequence.recorded_path = "recorded.wav"
    SequenceWidgetAnalysisOpsMixin._capture_excel_export_cache(sequence)

    assert sequence.data_struct.analysis_result_dict == {
        "item--通道1": (True, 0.0), "item--通道3": (True, 0.0),
    }
    cached = sequence._excel_export_cache["analysis_items_data"]
    assert list(cached) == ["item--通道1", "item--通道3"]
    assert [entry["raw_channel"] for entry in cached.values()] == [0, 2]
    assert all(entry["config_key"] == "item" for entry in cached.values())
    assert all(entry["multi_channel_expansion"] for entry in cached.values())


def test_partly_missing_recorded_item_runs_valid_channel_and_keeps_runtime_name():
    sequence, loader, messages, _events, created = build_sequence(
        factor_map={0: 2.5},
        active_input_channels=[0],
        items=[("item", "SPL", {"analysis_channels": [0, 7]})],
    )

    assert sequence._run_analysis_impl(show_windows=False) is True

    loader.assert_called_once_with(DEVICE)
    assert [instance.key for instance in created] == ["item--通道1"]
    assert created[0]._sequence_multi_channel_expansion is True
    assert created[0]._sequence_window_key == "item--通道1"
    assert list(sequence._analysis_preflight_skips) == ["item--通道8"]
    messages.warning.assert_called_once()
    assert "In8" in messages.warning.call_args.args[2]


def test_all_recorded_channels_missing_does_not_load_calibration_or_analyze():
    sequence, loader, messages, _events, created = build_sequence(
        active_input_channels=[0],
        items=[("item", "SPL", {"analysis_channels": [2, 7]})],
    )

    assert sequence._run_analysis_impl(show_windows=False) is False

    assert created == []
    loader.assert_not_called()
    sequence._capture_excel_export_cache.assert_not_called()
    messages.warning.assert_called_once()


def test_single_recorded_channel_keeps_legacy_window_key():
    sequence, _loader, _messages, _events, created = build_sequence(
        factor_map={2: 2.5},
        active_input_channels=[2],
        items=[("item", "SPL", {"analysis_channels": [2]})],
    )

    sequence._run_analysis_impl(show_windows=False)

    assert len(created) == 1
    assert created[0]._sequence_window_key == "item"
    assert created[0]._sequence_multi_channel_expansion is False


def test_expanded_windows_restore_and_register_distinct_geometry_keys(monkeypatch):
    sequence, _loader, _messages, _events, created = build_sequence(
        factor_map={0: 1.0, 2: 1.0},
        active_input_channels=[0, 2],
        items=[("item", "SPL", {"analysis_channels": [0, 2]})],
    )
    for method in ("setMinimumSize", "setGeometry", "installEventFilter", "show"):
        monkeypatch.setattr(FakeAnalysis, method, mock.Mock(), raising=False)
    sequence._analysis_window_key_by_obj = {}
    sequence._maybe_show_analysis_result_summary = mock.Mock()
    sequence._analysis_window_display_geometry = mock.Mock(
        side_effect=lambda _key, default, **_kwargs: default
    )

    sequence._run_analysis_impl(show_windows=True)

    assert [call.args[0] for call in sequence._analysis_window_display_geometry.call_args_list] == [
        "item--通道1", "item--通道3"
    ]
    assert sequence._analysis_window_key_by_obj == {
        created[0]: "item--通道1", created[1]: "item--通道3"
    }


@pytest.mark.parametrize("limit_metric", ["curve_y", "overall_spl"])
def test_real_spl_runtime_reads_distinct_recorded_columns_and_judges_each_channel(monkeypatch, limit_metric):
    from ui.signal_analysis_window import Spl

    app = QApplication.instance() or QApplication([])
    sequence, _loader, _messages, _events, _created = build_sequence(
        factor_map={2: 2.0, 7: 1.0},
        active_input_channels=[7, 2],
        items=[("item", "SPL", {
            "analysis_channels": [2, 7],
            "weighting": "Z",
            "smooth_checked": False,
            "show_overall_spl": True,
            "limit_checked": True,
            "limit_metric": limit_metric,
            "scalar_upper_enabled": True,
            "scalar_upper_value": 70.0,
            "limit_mode": "manual",
            "manual_input_mode": "constant",
            "constant_upper_enabled": True,
            "constant_lower_enabled": False,
            "constant_upper_value": 70.0,
        })],
    )
    monkeypatch.setattr(
        sequence,
        "_get_legacy_analysis_class_mapping",
        lambda: {"SPL": Spl},
    )
    sequence.data_struct.store_wave_data_multi = np.column_stack([
        np.full(2401, 0.2), np.full(2401, 0.01),
    ])
    sequence.data_struct.store_wave_data = np.full(2401, 99.0)
    sequence.data_struct.sample_rate = 48000

    try:
        assert sequence._run_analysis_impl(show_windows=False) is True
        first, second = sequence.analysis_window
        assert first.result["recorded_signal"] == pytest.approx(np.full(2401, 0.01))
        assert second.result["recorded_signal"] == pytest.approx(np.full(2401, 0.2))
        assert first.result["overall_spl"] == pytest.approx(60.0)
        assert second.result["overall_spl"] == pytest.approx(80.0)
        results = sequence.data_struct.analysis_result_dict
        assert results["item--通道3"][0] is True
        assert results["item--通道8"][0] is False
        assert SequenceWidgetStreamingOpsMixin._summarize_ok_ng(sequence) == (False, "NG")
    finally:
        for instance in sequence.analysis_window:
            instance.close()
        app.processEvents()


def test_real_imported_spl_overall_judgment_uses_one_wav_column_and_file_calibration(monkeypatch):
    from ui.signal_analysis_window import Spl

    app = QApplication.instance() or QApplication([])
    sequence, loader, messages, _events, _created = build_sequence(
        import_audio=True,
        legacy_factor=99.0,
        loader_error=MicCalibrationFormatError("must not load"),
        wav_metadata={
            "recorded_channels": [{
                "wav_channel_index": 1,
                "v2pa_factor": 2.0,
                "standard_spl": 94.0,
                "calibrated": True,
            }],
        },
        items=[("item", "SPL", {
            "analysis_channel": 1,
            "analysis_channels": [2, 7],
            "weighting": "Z",
            "limit_checked": True,
            "limit_metric": "overall_spl",
            "scalar_upper_value": 65.0,
            "show_overall_spl": False,
        })],
    )
    monkeypatch.setattr(
        sequence,
        "_get_legacy_analysis_class_mapping",
        lambda: {"SPL": Spl},
    )
    sequence.data_struct.store_wave_data_multi = np.column_stack([
        np.full(2401, 0.2), np.full(2401, 0.01),
    ])
    sequence.data_struct.store_wave_data = np.full(2401, 99.0)
    sequence.data_struct.sample_rate = 48000

    try:
        assert sequence._run_analysis_impl(show_windows=False) is True
        assert len(sequence.analysis_window) == 1
        instance = sequence.analysis_window[0]
        assert instance.analysis_config["analysis_channel"] == 1
        assert instance.v2pa_factor == 2.0
        assert instance.result["overall_spl"] == pytest.approx(60.0)
        assert sequence.data_struct.analysis_result_dict["item--通道2"][0] is True
        loader.assert_not_called()
        messages.warning.assert_not_called()
    finally:
        for instance in sequence.analysis_window:
            instance.close()
        app.processEvents()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"recorded_channels": "malformed"},
        {
            "recorded_channels": [
                {
                    "wav_channel_index": 0,
                    "v2pa_factor": 2.5,
                    "standard_spl": 94.0,
                    "calibrated": True,
                }
            ]
        },
        {
            "recorded_channels": [
                {
                    "wav_channel_index": 1,
                    "v2pa_factor": None,
                    "standard_spl": None,
                    "calibrated": False,
                }
            ]
        },
    ],
    ids=["absent", "malformed", "missing-record", "uncalibrated"],
)
def test_imported_wav_factor_fallback_warns_once_and_executes_valid_items(metadata):
    sequence, loader, messages, events, created = build_sequence(
        import_audio=True,
        legacy_factor=99.0,
        wav_metadata=metadata,
        items=[
            ("spl", "SPL", {"analysis_channel": 1}),
            ("spec", "Spec", {"analysis_channel": 1}),
            ("fba", "FBA", {"analysis_channel": 1}),
        ],
    )
    sequence.data_struct.store_wave_data_multi = np.zeros((8, 2))

    assert sequence._run_analysis_impl(show_windows=False) is True

    loader.assert_not_called()
    messages.warning.assert_called_once_with(
        sequence,
        "音频校准数据缺失",
        "该音频文件未包含有效校准数据，分析结果仅供参考。",
    )
    assert [instance.v2pa_factor for instance in created] == [1.0, 1.0, 1.0]
    assert [event[0] for event in events] == [
        "warning",
        "calculate",
        "calculate_spec",
        "calculate_fba",
    ]


def test_imported_wav_warning_type_error_propagates_without_marking_shown():
    sequence, _loader, messages, _events, _created = build_sequence(
        import_audio=True,
    )
    messages.warning.side_effect = TypeError("invalid Qt parent")

    with pytest.raises(TypeError, match="invalid Qt parent"):
        sequence._show_imported_wav_calibration_warning(True)

    assert sequence.data_struct.wav_calibration_warning_shown is False

    messages.warning.side_effect = None
    sequence._show_imported_wav_calibration_warning(True)

    assert messages.warning.call_count == 2
    assert sequence.data_struct.wav_calibration_warning_shown is True


def test_imported_warning_order_skips_missing_channel_and_rewarns_next_run():
    sequence, loader, messages, events, _created = build_sequence(
        import_audio=True,
        legacy_factor=99.0,
        items=[
            ("missing", "SPL", {"analysis_channel": 2}),
            ("valid", "Spec", {"analysis_channel": 1}),
        ],
    )
    sequence.data_struct.store_wave_data_multi = np.zeros((8, 2))

    assert sequence._run_analysis_impl(show_windows=False) is True
    assert [event[0] for event in events] == [
        "warning",
        "warning",
        "calculate_spec",
    ]
    assert messages.warning.call_args_list[1].args[2] == (
        "该音频文件未包含有效校准数据，分析结果仅供参考。"
    )
    assert sequence._imported_wav_channel_v2pa_factors == {"valid": 1.0}

    events.clear()
    assert sequence._run_analysis_impl(show_windows=False) is True
    assert [event[0] for event in events] == [
        "warning",
        "warning",
        "calculate_spec",
    ]
    assert messages.warning.call_count == 4
    loader.assert_not_called()


def test_imported_all_missing_channels_only_emit_preflight_warning():
    resolver = mock.Mock(side_effect=resolve_wav_channel_v2pa_factor)
    sequence, loader, messages, events, created = build_sequence(
        import_audio=True,
        wav_resolver=resolver,
        items=[
            ("missing-spl", "SPL", {"analysis_channel": 2}),
            ("missing-fba", "FBA", {"analysis_channel": 3}),
        ],
    )
    sequence.data_struct.store_wave_data_multi = np.zeros((8, 2))

    assert sequence._run_analysis_impl(show_windows=False) is False

    messages.warning.assert_called_once()
    assert messages.warning.call_args.args[1] == "分析通道不存在"
    assert [event[0] for event in events] == ["warning"]
    assert sequence._imported_wav_channel_v2pa_factors == {}
    assert created == []
    resolver.assert_not_called()
    loader.assert_not_called()


def test_preflight_warns_once_before_calibration_and_only_runs_valid_items():
    sequence, loader, messages, events, created = build_sequence(
        factor_map={0: 2.5, 2: 7.0},
        active_input_channels=[0, 2],
        items=[
            ("missing-spec", "Spec", {"analysis_channel": 1}),
            ("valid-spl", "SPL", {"analysis_channel": 2}),
            ("missing-fba", "FBA", {"analysis_channel": 3}),
        ],
    )
    original_prepare = sequence._prepare_live_mic_calibration_batch

    def prepare():
        events.append(("prepare_calibration",))
        return original_prepare()

    sequence._prepare_live_mic_calibration_batch = prepare

    assert sequence._run_analysis_impl(show_windows=False) is True

    loader.assert_called_once_with(DEVICE)
    messages.warning.assert_called_once()
    assert [event[0] for event in events] == [
        "warning",
        "prepare_calibration",
        "calculate",
    ]
    body = messages.warning.call_args.args[2]
    assert "missing-spec" in body and "请求 In2" in body
    assert "missing-fba" in body and "请求 In4" in body
    assert "可用 In1、In3" in body
    assert [instance.key for instance in created] == ["valid-spl--通道3"]
    assert created[0].analysis_config["analysis_channel"] == 1
    assert list(sequence._analysis_preflight_skips) == [
        "missing-spec",
        "missing-fba",
    ]
    assert sequence.data_struct.analysis_result_dict == {}
    sequence._capture_excel_export_cache.assert_called_once()
    sequence._maybe_export_excel_results.assert_called_once()


def test_all_preflight_items_skipped_returns_false_retains_audio_and_rewarns():
    sequence, loader, messages, events, created = build_sequence(
        active_input_channels=[0],
        items=[
            ("missing-spl", "SPL", {"analysis_channel": 1}),
            ("missing-fba", "FBA", {"analysis_channel": 2}),
        ],
    )
    audio = np.asarray([[0.1], [-0.1]], dtype=np.float32)
    sequence.data_struct.store_wave_data_multi = audio

    assert sequence._run_analysis_impl(show_windows=False) is False
    assert sequence._run_analysis_impl(show_windows=False) is False

    assert messages.warning.call_count == 2
    assert loader.call_count == 0
    assert created == []
    assert not any(event[0] == "calculate" for event in events)
    assert sequence.data_struct.store_wave_data_multi is audio
    sequence._capture_excel_export_cache.assert_not_called()
    sequence._maybe_export_excel_results.assert_not_called()
    assert sequence._analysis_preflight_warning_shown is True


def test_preflight_returns_false_when_only_remaining_item_is_not_executable():
    sequence, loader, _messages, _events, created = build_sequence(
        active_input_channels=[0],
        items=[
            ("missing-spl", "SPL", {"analysis_channel": 1}),
            ("future", "FUTURE", {}),
        ],
    )

    assert sequence._run_analysis_impl(show_windows=False) is False
    loader.assert_not_called()
    assert created == []
    sequence._capture_excel_export_cache.assert_not_called()


def test_rsc_does_not_enter_microphone_factor_resolution():
    sequence, _loader, _messages, _events, _created = build_sequence()
    sequence._live_mic_channel_v2pa_factors = {}
    sequence._missing_mic_calibration_channels = []
    sequence._missing_mic_calibration_channel_set = set()

    sequence.instance_analysis_class(
        "reference", "RSC", {"analysis_channel": 2}
    )

    instance = sequence.analysis_window[0]
    assert not hasattr(instance, "v2pa_factor")
    assert instance.analysis_config["analysis_channel"] == 2
    assert sequence._missing_mic_calibration_channels == []


def test_lp_keeps_legacy_channel_coercion_outside_preflight_scope():
    sequence, _loader, _messages, _events, _created = build_sequence(
        active_input_channels=[0, 1]
    )
    sequence._live_mic_channel_v2pa_factors = {1: 4.0}
    sequence._missing_mic_calibration_channels = []
    sequence._missing_mic_calibration_channel_set = set()

    sequence.instance_analysis_class("loose", "LP", {"analysis_channel": True})

    instance = sequence.analysis_window[0]
    assert instance.analysis_config["analysis_channel"] == 1


def test_rsc_only_batch_never_loads_microphone_calibration_and_still_executes():
    sequence, loader, messages, events, created = build_sequence(
        loader_error=MicCalibrationFormatError("must not load"),
        items=[("reference", "RSC", {"analysis_channel": 2})],
    )

    sequence._run_analysis_impl(show_windows=False)

    loader.assert_not_called()
    messages.critical.assert_not_called()
    messages.warning.assert_not_called()
    assert len(created) == 1
    assert events == [("calculate_rsc", "reference")]


def test_mixed_batch_still_loads_once_and_aborts_before_rsc_or_spl_execution():
    sequence, loader, messages, events, created = build_sequence(
        loader_error=MicCalibrationIOError("denied"),
        items=[
            ("reference", "RSC", {"analysis_channel": 0}),
            ("spl", "SPL", {"analysis_channel": 0}),
        ],
    )

    sequence._run_analysis_impl(show_windows=False)

    loader.assert_called_once_with(DEVICE)
    messages.critical.assert_called_once()
    assert created == []
    assert events == [
        (
            "critical",
            "输入校准文件错误",
            "输入校准文件错误，本次分析已停止",
        )
    ]


def test_channel_mismatch_remains_authoritative_when_factor_is_missing():
    sequence, _loader, _messages, _events, _created = build_sequence(
        active_input_channels=[0, 2]
    )
    sequence._live_mic_channel_v2pa_factors = {}
    sequence._missing_mic_calibration_channels = []
    sequence._missing_mic_calibration_channel_set = set()

    sequence.instance_analysis_class(
        "mismatch", "SPL", {"analysis_channel": 1}
    )

    instance = sequence.analysis_window[0]
    assert instance.v2pa_factor == 1.0
    assert instance._channel_mismatch is True
    assert instance._channel_mismatch_info == {
        "raw_channel": 1,
        "active_input_channels": [0, 2],
    }
    assert instance.analysis_config["analysis_channel"] == 0


class RuntimeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)


def _enable_real_ok_ng_runtime(sequence):
    sequence._can_output_ok_ng = (
        SequenceWidgetStreamingOpsMixin._can_output_ok_ng.__get__(sequence)
    )
    sequence._summarize_ok_ng = mock.Mock(
        wraps=SequenceWidgetStreamingOpsMixin._summarize_ok_ng.__get__(sequence)
    )
    sequence.count_board = SimpleNamespace(
        mode="test",
        set_test_result_file=mock.Mock(),
        set_test_text=mock.Mock(),
    )
    sequence.data_btn = RuntimeButton()
    sequence.replayer_btn = RuntimeButton()
    sequence._awaiting_ok_ng = True
    sequence._sn_clear_on_next_scan = True
    sequence.update_player_btn_is_paused = mock.Mock()
    sequence._is_directional_cycle_active = lambda: False
    sequence._is_manual_product_condition_cycle_active = lambda: False
    sequence._finalize_test_run = mock.Mock()


@pytest.mark.parametrize("channel_config", [
    {"analysis_channel": 1},
    {"analysis_channels": [1, 2]},
])
def test_preflight_skipped_judging_item_cannot_finalize_non_judging_batch(channel_config):
    sequence, _loader, messages, events, _created = build_sequence(
        factor_map={0: 2.5},
        active_input_channels=[0],
        items=[
            ("skipped-spl", "SPL", {**channel_config, "limit_checked": True}),
            ("valid-spec", "Spec", {"analysis_channel": 0}),
        ],
    )
    _enable_real_ok_ng_runtime(sequence)

    assert sequence._run_analysis_impl(show_windows=False) is True

    assert ("calculate_spec", "valid-spec--通道1") in events
    sequence._summarize_ok_ng.assert_not_called()
    sequence._finalize_test_run.assert_not_called()
    sequence.count_board.set_test_result_file.assert_not_called()
    assert sequence.data_struct.analysis_result_dict == {}
    assert any("无法产出 OK/NG" in call.args[2] for call in messages.warning.call_args_list)


@pytest.mark.parametrize("channel_config", [
    {"analysis_channel": 0},
    {"analysis_channels": [0, 2]},
])
def test_preflight_keeps_finalization_when_executable_judging_item_remains(channel_config):
    sequence, _loader, _messages, _events, _created = build_sequence(
        factor_map={0: 2.5},
        active_input_channels=[0],
        items=[
            ("skipped-fba", "FBA", {"analysis_channel": 1, "limit_checked": True}),
            (
                "valid-spl",
                "SPL",
                {
                    **channel_config,
                    "limit_checked": True,
                    "_test_judgment": True,
                },
            ),
        ],
    )
    _enable_real_ok_ng_runtime(sequence)

    assert sequence._run_analysis_impl(show_windows=False) is True

    sequence._summarize_ok_ng.assert_called_once_with()
    sequence._finalize_test_run.assert_called_once_with(
        "OK",
        update_recent_session=True,
    )
    sequence.count_board.set_test_result_file.assert_called_once_with("OK")


class RuntimeAnalysis:
    def __init__(self, key, events):
        self.key = key
        self.events = events

    def calculate_spl(self):
        self.events.append("calculate")
        return True


class StreamingAnalysisHost(
    SequenceWidgetAnalysisOpsMixin,
    SequenceWidgetSerialTriggerOpsMixin,
    SequenceWidgetStreamingOpsMixin,
):
    def __init__(self):
        self.events = []
        self.report_updates = []
        self.sequence_config = [
            {"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {}}}}
        ]
        self.analysis_config = {
            "display_sequence": ["spl"],
            "spl": {"type": "SPL", "analysis_channel": 0},
        }
        self.product_test_pdf_report_config = {"enabled": True}
        self.product_test_condition_configs = [
            {
                "key": "condition-1",
                "trigger_state": "AAAA",
                "condition_name": "Condition 1",
            },
            {
                "key": "condition-2",
                "trigger_state": "BBBB",
                "condition_name": "Condition 2",
            },
        ]
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = "group-1"
        self._displayed_manual_product_condition_group_id = "group-1"
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = "condition-1"
        self._active_product_condition_config = dict(
            self.product_test_condition_configs[0]
        )
        self._waveform_display_override_direction = "condition-1"
        self._current_trigger_direction = "condition-1"
        self._current_recent_session_id = "session-1"
        self.mic = DEVICE
        self._active_input_channels = [0]
        self.v2pa_factor = 8.0
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.data_struct = SimpleNamespace(
            store_wave_data=None,
            store_wave_data_multi=None,
            sample_rate=48000,
            analysis_result_dict={},
        )
        self.count_board = SimpleNamespace(mode="view")
        self.recorded_path = "recorded.wav"
        self.recorded_signal_info = {"labels": "not_labeled"}
        self.streaming_wav_writer = None
        self.streaming_processor = None
        self.streaming_stimulus_data = None
        self.streaming_mode = None
        self.player_status_flag = True
        self.clicked_player_flag = True
        self._record_workflow_busy = True
        self._serial_product_condition_executing = True
        self._serial_product_session_started = True
        self._serial_product_latched_frame = "AAAA"
        self._serial_product_waiting_for_close = True
        self._serial_product_pending_close_frame = "CCCC"
        self._queued_directional_trigger = "queued"
        self._pending_serial_trigger_direction = "pending"
        self._awaiting_ok_ng = True
        self._sn_clear_on_next_scan = True
        self._pending_recent_session_append = True
        self.count_updates = 0
        self.started = []
        self.data_btn = RuntimeButton()
        self.replayer_btn = RuntimeButton()
        self.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
        self.default_logger = SimpleNamespace(
            error=mock.Mock(), info=mock.Mock(), warning=mock.Mock()
        )
        self.left_panel = SimpleNamespace(
            set_condition_result=mock.Mock(),
            set_current_stage=mock.Mock(),
            set_final_result=mock.Mock(),
        )

    def screen(self):
        size = SimpleNamespace(width=lambda: 1600, height=lambda: 900)
        return SimpleNamespace(size=lambda: size)

    def instance_analysis_class(self, key, _item_type, params):
        instance = RuntimeAnalysis(key, self.events)
        instance.data_struct = self.data_struct
        instance.analysis_config = dict(params)
        instance.v2pa_factor = self._resolve_live_mic_channel_v2pa_factor(0)
        instance._sequence_analysis_key = key
        instance._channel_mismatch = False
        instance._channel_mismatch_info = None
        self.analysis_window.append(instance)

    def _close_analysis_windows(self):
        self.analysis_window.clear()

    def _resolve_recording_acq_detail(self):
        return {}

    def plot_waveform_to_workspace(self, *_args, **_kwargs):
        return None

    def _clear_active_recording_direction(self):
        return None

    def _update_current_recent_session_result(self, *_args, **_kwargs):
        return None

    def _is_manual_product_condition_cycle_active(self):
        return True

    def _mark_manual_product_condition_recording_completed(self):
        self.events.append("mark_complete")
        return super()._mark_manual_product_condition_recording_completed()

    def _update_manual_product_mark_group_count(self, *_args):
        self.count_updates += 1
        return True

    def _finalize_serial_product_condition_after_analysis(self):
        self.events.append("finalize_condition")
        return super()._finalize_serial_product_condition_after_analysis()

    def _advance_manual_product_condition_cycle_after_recording(self):
        self.events.append("advance_condition")

    def _on_serial_product_condition_completed(self):
        self.events.append("condition_completed")
        return super()._on_serial_product_condition_completed()

    def _product_condition_sequence(self):
        return [dict(item) for item in self.product_test_condition_configs]

    def _prepare_next_manual_product_condition_recording(self):
        condition = self.product_test_condition_configs[
            self._manual_product_condition_index
        ]
        self._active_product_condition_key = condition["key"]
        self._active_product_condition_config = dict(condition)
        return True

    def start_this_play(self, _label):
        self.started.append(self._active_product_condition_key)
        self._record_workflow_busy = True
        self.player_status_flag = True

    def _refresh_current_manual_product_final_from_group(self, *_args):
        return None

    def update_player_btn_is_paused(self):
        self.events.append("ui_cleanup")

    def _reset_barcode_commit_dedup(self):
        return None

    def _drain_queued_directional_trigger(self):
        self.events.append("drain_trigger")

    def _capture_excel_export_cache(self):
        return None

    def _maybe_export_excel_results(self):
        return None

    def _can_output_ok_ng(self):
        return False, ""

    def _sync_left_panel_analysis_details(self, *_args):
        return None

    def _hide_analysis_window(self, _instance):
        return None

    def _capture_current_analysis_report_snapshot(self, session_id=None):
        self.events.append(("report_snapshot", session_id))

    def _update_recent_session(self, session_id, **fields):
        self.report_updates.append((session_id, fields))


def run_streaming_completion(host, factor_loader):
    host.channel_workspace = SimpleNamespace(
        all_subwindows=lambda: [SimpleNamespace(channel_index=0)]
    )
    host._project_normalized_waveform_to_workspace = mock.Mock()
    recording_manager = SimpleNamespace(
        save_signal_info_to_db=lambda *_args: (error_code.OK, "saved")
    )
    with mock.patch(
        "ui.sequence.sequence_widget_analysis_ops.load_mic_channel_v2pa_factors",
        factor_loader,
    ), mock.patch(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox.critical"
    ) as critical, mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.resolve_startup_trim_samples",
        return_value=0,
    ), mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.validate_recorded_audio",
        return_value=(True, "", {}),
    ), mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.RecordingManager",
        return_value=recording_manager,
    ):
        host._on_streaming_complete(
            recorded_mono=np.asarray([0.1, -0.1], dtype=np.float32),
            recorded_multi=np.asarray([[0.1], [-0.1]], dtype=np.float32),
            sample_rate=48000,
            completion_source="test",
        )
    return critical


def test_streaming_completion_queues_analysis_and_advances_recording_workflow():
    host = StreamingAnalysisHost()
    loader = mock.Mock(
        side_effect=AssertionError("recording completion must not run analysis")
    )
    enqueue = mock.Mock(return_value=True)
    host._enqueue_automatic_analysis_current_recording = enqueue

    critical = run_streaming_completion(host, loader)

    loader.assert_not_called()
    critical.assert_not_called()
    enqueue.assert_called_once_with()
    assert "calculate" not in host.events
    assert "finalize_condition" in host.events
    assert "advance_condition" in host.events
    assert "condition_completed" in host.events
    assert host._manual_product_condition_completed_keys == {"condition-1"}
    assert host.count_updates == 1
    assert host._serial_product_condition_executing is False
    assert host._serial_product_session_started is False
    assert ("report_snapshot", "session-1") not in host.events
    assert host._record_workflow_busy is False
    assert host.events[-2:] == ["ui_cleanup", "drain_trigger"]
    assert any(
        "recording completed successfully" in call.args[0]
        for call in host.default_logger.info.call_args_list
    )


def test_streaming_completion_without_analysis_keeps_legacy_completion_path():
    host = StreamingAnalysisHost()
    host._is_manual_product_condition_cycle_active = lambda: False
    host._should_run_silent_analysis_after_recording = lambda: False
    loader = mock.Mock(side_effect=AssertionError("analysis must not run"))

    critical = run_streaming_completion(host, loader)

    loader.assert_not_called()
    critical.assert_not_called()
    assert "mark_complete" not in host.events
    assert host.count_updates == 0
    assert "finalize_condition" in host.events
    assert "advance_condition" in host.events
    assert "condition_completed" in host.events
    assert host._serial_product_condition_executing is False
    assert host._record_workflow_busy is False
    assert any(
        "recording completed successfully" in call.args[0]
        for call in host.default_logger.info.call_args_list
    )
