import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import numpy as np

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


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_OPS = ROOT / "ui" / "sequence" / "sequence_widget_analysis_ops.py"
DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 4,
}


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
            "_run_analysis_impl",
            "instance_analysis_class",
        ],
        {
            "get_class_mapping": lambda: {
                "SPL": make_analysis,
                "RSC": make_rsc,
            },
            "load_mic_channel_v2pa_factors": loader,
            "MicCalibrationFormatError": MicCalibrationFormatError,
            "MicCalibrationIOError": MicCalibrationIOError,
            "QMessageBox": messages,
            "extract_ai_runtime_state": lambda *_args: {
                "has_ai_analysis": False,
                "scores": {},
            },
        },
    )
    sequence = cls()
    sequence.mic = DEVICE
    sequence.v2pa_factor = legacy_factor
    sequence._active_input_channels = list(active_input_channels or [0])
    sequence._is_import_audio_mode = lambda: import_audio
    sequence.data_struct = SimpleNamespace(analysis_result_dict={})
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


def test_imported_audio_never_loads_microphone_calibration_and_preserves_factor():
    sequence, loader, messages, _events, created = build_sequence(
        loader_error=MicCalibrationFormatError("must not load"),
        import_audio=True,
        legacy_factor=7.25,
        items=[("spl", "SPL", {"analysis_channel": 3})],
    )

    sequence._run_analysis_impl(show_windows=False)

    loader.assert_not_called()
    messages.warning.assert_not_called()
    messages.critical.assert_not_called()
    assert created[0].v2pa_factor == 7.25
    assert created[0].analysis_config["analysis_channel"] == 0


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


@pytest.mark.parametrize(
    "error",
    [MicCalibrationFormatError("corrupt"), MicCalibrationIOError("denied")],
)
def test_streaming_completion_stops_automation_and_reports_calibration_failure(error):
    host = StreamingAnalysisHost()
    loader = mock.Mock(side_effect=error)

    critical = run_streaming_completion(host, loader)

    loader.assert_called_once_with(DEVICE)
    critical.assert_called_once()
    assert "finalize_condition" not in host.events
    assert "advance_condition" not in host.events
    assert "condition_completed" not in host.events
    assert host._manual_product_condition_completed_keys == set()
    assert host.count_updates == 0
    assert host._serial_product_condition_executing is False
    assert host._serial_product_session_started is False
    assert host._serial_product_waiting_for_close is False
    assert host._serial_product_pending_close_frame == ""
    assert host._queued_directional_trigger == ""
    assert host._pending_serial_trigger_direction == ""
    assert host._active_product_condition_key == ""
    assert host._awaiting_ok_ng is False
    assert host._sn_clear_on_next_scan is False
    assert host._pending_recent_session_append is False
    assert host.report_updates[0][0] == "session-1"
    assert host.report_updates[0][1]["analysis_report_state"] == "failed"
    assert str(error) in host.report_updates[0][1]["analysis_report_items"][0]["error"]
    assert host._record_workflow_busy is False
    assert host.events[-2:] == ["ui_cleanup", "drain_trigger"]
    assert not any(
        "recording completed successfully" in call.args[0]
        for call in host.default_logger.info.call_args_list
    )

    host.on_serial_full_frame_received(
        {"raw_hex": "AAAA", "product_full_frame": True}
    )

    assert host.started == ["condition-1"]
    assert host._serial_product_condition_executing is True
    assert host.report_updates[0][1]["analysis_report_state"] == "failed"


def test_streaming_completion_success_still_advances_automatic_workflow():
    host = StreamingAnalysisHost()
    loader = mock.Mock(return_value={0: 2.5})

    critical = run_streaming_completion(host, loader)

    loader.assert_called_once_with(DEVICE)
    critical.assert_not_called()
    assert "calculate" in host.events
    assert "finalize_condition" in host.events
    assert "advance_condition" in host.events
    assert "condition_completed" in host.events
    assert host._manual_product_condition_completed_keys == {"condition-1"}
    assert host.count_updates == 1
    assert host._serial_product_condition_executing is False
    assert host._serial_product_session_started is False
    assert ("report_snapshot", "session-1") in host.events
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
