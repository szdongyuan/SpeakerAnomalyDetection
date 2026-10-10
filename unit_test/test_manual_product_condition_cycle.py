import logging
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class _SpyLeftPanel:
    def __init__(self, conditions):
        self.conditions = conditions
        self.condition_results = []
        self.final_results = []
        self.stages = []
        self.analysis_details = []
        self.automatic_round_result = None

    def set_condition_result(self, condition, result_text, tone=None):
        self.condition_results.append((condition, result_text, tone))

    def reset_result_panel(self):
        for condition in self.conditions:
            self.set_condition_result(condition["key"], "待检测", "pending")
        self.set_final_result("待判定", "pending")

    def set_final_result(self, result_text, tone=None):
        self.final_results.append((result_text, tone))

    def set_current_stage(self, stage_text, tone=None):
        self.stages.append((stage_text, tone))

    def set_condition_analysis_details(self, condition, detail_values):
        self.analysis_details.append((condition, dict(detail_values or {})))
        return True

    def get_automatic_round_result(self):
        return self.automatic_round_result


class _SpyCountBoard:
    def __init__(self):
        self.mode = "test"
        self.test_results = []
        self.mark_results = []
        self.mark_relabels = []
        self.refreshes = 0

    def set_test_result_file(self, label):
        self.test_results.append(label)

    def set_test_text(self):
        self.refreshes += 1

    def append_mark_result_file(self, label):
        self.mark_results.append(label)

    def update_mark_result_file_on_relabel(self, old_label, new_label):
        self.mark_relabels.append((old_label, new_label))

    def set_mark_text(self):
        self.refreshes += 1


class _DummyManualCycleWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.default_logger = logging.getLogger(__name__)
        self.product_test_condition_configs = [
            {"key": "q6000", "condition_name": "6000", "test_queue": "queue_6000"},
            {"key": "q7000", "condition_name": "7000", "test_queue": "queue_7000"},
            {"key": "q8000", "condition_name": "8000", "test_queue": "queue_8000"},
        ]
        self.loaded_queues = []
        self.started = []
        self.cleared_waveforms = 0
        self.clicked_player_flag = False
        self.sequence_config = []
        self.analysis_config = {}
        self.count_board = _SpyCountBoard()
        self.left_panel = _SpyLeftPanel(self.product_test_condition_configs)
        self.channel_workspace = SimpleNamespace(results=[])
        self.channel_workspace.set_condition_result = lambda key, label: self.channel_workspace.results.append((key, label))
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""
        self._current_cycle_recorded_count = None
        self._current_run_recording_token = ""
        self.last_play_count = None
        self._token_seq = 0

    def _load_sequence_config_for_product_condition(self, condition_config):
        queue_name = condition_config["test_queue"]
        self.loaded_queues.append(queue_name)
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
                        "detail": {"sample_rate": 44100},
                    },
                    "analysis_list": {},
                }
            }
        ]
        return True, ""

    def _generate_recording_token(self):
        self._token_seq += 1
        return f"token_{self._token_seq}"

    _generate_product_condition_group_id = _generate_recording_token

    def clear_all_direction_waveforms(self):
        self.cleared_waveforms += 1

    def start_this_play(self, label="not_labeled"):
        self.started.append((label, self._active_product_condition_key, self._resolve_recording_name_suffix()))




class TestManualProductConditionCycle(unittest.TestCase):


    def test_manual_and_unified_product_results_share_one_summary(self):
        widget = _DummyManualCycleWidget()
        widget._manual_product_condition_group_id = "group-1"
        widget._manual_product_condition_results = {
            "q6000": "OK",
            "q7000": "NG",
            "q8000": "OK",
        }
        widget._manual_product_condition_completed_keys = {
            "q6000",
            "q7000",
            "q8000",
        }

        self.assertEqual(
            widget._product_group_result_state("group-1"),
            (True, "NG"),
        )
        self.assertEqual(
            widget._manual_product_group_result_state("group-1"),
            (True, "NG"),
        )

    def test_play_button_cycles_through_product_conditions(self):
        widget = _DummyManualCycleWidget()

        widget.on_clicked_player_btn()
        self.assertEqual(widget.loaded_queues, ["queue_6000"])
        self.assertEqual(widget.started[-1], ("not_labeled", "q6000", "_6000"))
        first_group_id = widget._manual_product_condition_group_id
        self.assertTrue(first_group_id)
        self.assertEqual(widget._current_cycle_recorded_count, first_group_id)
        self.assertEqual(widget.cleared_waveforms, 1)

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget.loaded_queues[-1], "queue_7000")
        self.assertEqual(widget.started[-1], ("not_labeled", "q7000", "_7000"))
        self.assertEqual(widget._manual_product_condition_group_id, first_group_id)

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget.loaded_queues[-1], "queue_8000")
        self.assertEqual(widget.started[-1], ("not_labeled", "q8000", "_8000"))
        self.assertEqual(widget._manual_product_condition_group_id, first_group_id)

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget.loaded_queues[-1], "queue_6000")
        self.assertEqual(widget.started[-1], ("not_labeled", "q6000", "_6000"))
        self.assertNotEqual(widget._manual_product_condition_group_id, first_group_id)
        self.assertEqual(widget.cleared_waveforms, 2)

    def test_manual_product_condition_runtime_key_prefers_explicit_key(self):
        widget = _DummyManualCycleWidget()
        widget.product_test_condition_configs = [
            {"key": "uuid_6000", "trigger_state": "01", "condition_name": "6000", "test_queue": "queue_6000"},
            {"key": "uuid_7000", "trigger_state": "02", "condition_name": "7000", "test_queue": "queue_7000"},
            {"key": "uuid_8000", "trigger_state": "03", "condition_name": "8000", "test_queue": "queue_8000"},
        ]

        self.assertEqual(
            [
                widget._product_condition_runtime_key(condition, index)
                for index, condition in enumerate(widget.product_test_condition_configs)
            ],
            ["uuid_6000", "uuid_7000", "uuid_8000"],
        )

    def test_play_button_with_complete_status_codes_respects_serial_enabled(self):
        for serial_enabled in (False, True):
            with self.subTest(serial_enabled=serial_enabled):
                widget = _DummyManualCycleWidget()
                widget.default_logger = logging.getLogger(__name__)
                for index, condition in enumerate(widget.product_test_condition_configs, 1):
                    condition["trigger_state"] = f"0{index}"
                widget._serial_trigger_config = {"enabled": serial_enabled}

                widget.on_clicked_player_btn()

                if serial_enabled:
                    self.assertEqual(widget.loaded_queues, [])
                    self.assertEqual(widget.started, [])
                    self.assertEqual(widget._manual_product_condition_group_id, "")
                else:
                    self.assertEqual(widget.loaded_queues, ["queue_6000"])
                    self.assertEqual(widget.started, [("not_labeled", "q6000", "_6000")])
                    self.assertTrue(widget._manual_product_condition_group_id)

    def test_mark_mode_allows_next_play_with_unlabeled_history(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"

        widget.on_clicked_player_btn()
        first_group_id = widget._manual_product_condition_group_id
        widget._mark_manual_product_condition_recording_completed()
        widget._advance_manual_product_condition_cycle_after_recording()
        widget.recent_test_sessions = ["recent_1"]
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": first_group_id,
                "condition_key": "q6000",
                "result_label": "not labeled",
                "recorded_signal_info": {"labels": "not_labeled"},
            }
        }

        with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning") as warning:
            widget.on_clicked_player_btn()

        warning.assert_not_called()
        self.assertEqual(widget.loaded_queues[-1], "queue_7000")
        self.assertEqual(widget.started[-1], ("not_labeled", "q7000", "_7000"))

    def test_product_condition_result_finalizes_after_all_conditions(self):
        widget = _DummyManualCycleWidget()

        widget.on_clicked_player_btn()
        self.assertIsNone(widget._update_manual_product_condition_result_after_analysis("OK"))
        self.assertEqual(widget.channel_workspace.results[-1], ("q6000", "OK"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertIsNone(widget._update_manual_product_condition_result_after_analysis("NG"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget._update_manual_product_condition_result_after_analysis("OK"), "NG")
        self.assertEqual(widget.left_panel.final_results[-1], ("NG", "ng"))

    def test_recent_session_update_refreshes_group_once(self):
        widget = _DummyManualCycleWidget()
        widget.recent_test_session_by_id = {
            "session-1": {"group_id": "group-1"}
        }
        refresh_calls = []
        widget._refresh_manual_product_condition_results_from_group = (
            lambda group_id: None
        )
        widget._refresh_current_manual_product_final_from_group = (
            lambda group_id: refresh_calls.append(group_id) or "OK"
        )

        widget._update_recent_session("session-1", result_label="OK")

        self.assertEqual(refresh_calls, ["group-1"])

    def test_recent_session_update_preserves_undisplayed_group_result(self):
        widget = _DummyManualCycleWidget()
        widget.recent_test_session_by_id = {
            "session-1": {"group_id": "history-group"}
        }
        widget._refresh_manual_product_condition_results_from_group = (
            lambda group_id: None
        )
        widget._refresh_current_manual_product_final_from_group = (
            lambda group_id: None
        )

        widget._update_recent_session("session-1", result_label="OK")

        self.assertEqual(widget.recent_test_session_by_id["session-1"]["result_label"], "OK")

    def test_left_panel_analysis_details_syncs_runtime_metrics(self):
        widget = _DummyManualCycleWidget()
        widget._active_product_condition_key = "q6000"
        widget.analysis_config = {
            "spl": {"type": "SPL", "weighting": "Z"},
            "loudness": {
                "type": "LOUD",
                "advanced": {"curve_y_unit": "sone"},
                "display": {
                    "summary_metrics": [
                        "steady_state_average_loudness",
                        "max_transient_loudness",
                        "specific_loudness_sum_sone",
                        "specific_loudness_summed_exceedance",
                    ]
                },
            },
            "ai": {"type": "AI"},
            "fba": {"type": "FBA"},
            "fft": {"type": "FFT"},
        }
        widget.data_struct = SimpleNamespace(analysis_result_dict={})
        widget.analysis_window = [
            SimpleNamespace(
                _sequence_analysis_key="spl",
                title_name="SPL--通道1",
                result={"overall_spl": 72.345},
                _get_spl_unit=lambda: "dB",
            ),
            SimpleNamespace(
                _sequence_analysis_key="loudness",
                title_name="响度--通道1",
                result={
                    "summary": {
                        "steady_state_average_sone": 4.2,
                        "max_transient_sone": 8.1,
                        "specific_loudness_sum_sone": 12.34,
                        "specific_loudness_summed_exceedance": 1.1629,
                    }
                },
                export_detail={},
            ),
            SimpleNamespace(
                _sequence_analysis_key="ai",
                title_name="AI--通道1",
                result="OK",
                export_detail={"label": "OK", "ok_score": 71.6, "ng_score": 28.4},
            ),
            SimpleNamespace(_sequence_analysis_key="fba", title_name="FBA--通道1"),
            SimpleNamespace(_sequence_analysis_key="fft", title_name="FFT--通道1"),
        ]
        widget.data_struct.analysis_result_dict = {
            "SPL--通道1": (True, 0.0),
            "响度--通道1": (True, 0.0),
            "FBA--通道1": (True, 0.0),
            "FFT--通道1": (False, 1.5),
        }

        synced = widget._sync_left_panel_analysis_details()

        self.assertTrue(synced)
        condition, detail_values = widget.left_panel.analysis_details[-1]
        self.assertEqual(condition, "q6000")
        self.assertEqual(detail_values["SPL"], "总体声压：72.34 dB；判定：OK")
        self.assertEqual(
            detail_values["响度"],
            "稳态平均响度：4.20 sone；最大瞬态响度：8.10 sone；"
            "特征响度总贡献：12.34 sone；特征响度超限总量：116.29 cSones；判定：OK",
        )
        self.assertNotIn("AI分析", detail_values)
        self.assertEqual(detail_values["FBA"], "OK")
        self.assertEqual(detail_values["FFT"], "NG")

    def test_spl_left_panel_fallback_keeps_target_distance_correction(self):
        widget = _DummyManualCycleWidget()
        instance = SimpleNamespace(
            result={
                "recorded_signal": [1.0, -1.0],
                "distance_correction_db": -20.0,
                "applied_correction_db": -25.0,
            },
            v2pa_factor=2.0,
            _get_spl_unit=lambda: "dB",
        )

        detail = widget._format_spl_left_panel_detail(
            instance,
            {"type": "SPL", "weighting": "Z"},
            "OK",
        )

        self.assertEqual(detail, "总体声压：75.00 dB；判定：OK")

    def test_left_panel_loudness_details_follow_display_metric_checks(self):
        widget = _DummyManualCycleWidget()
        widget._active_product_condition_key = "q6000"
        widget.analysis_config = {
            "loudness": {
                "type": "LOUD",
                "limit_checked": True,
                "limit_metric": "max_transient",
                "advanced": {"curve_y_unit": "sone"},
                "display": {
                    "summary_metrics": [
                        "steady_state_average_loudness",
                        "specific_loudness_summed_exceedance",
                    ]
                },
            },
        }
        widget.data_struct = SimpleNamespace(
            analysis_result_dict={"响度--通道1": (False, 1.5)}
        )
        widget.analysis_window = [
            SimpleNamespace(
                _sequence_analysis_key="loudness",
                title_name="响度--通道1",
                result={
                    "summary": {
                        "steady_state_average_sone": 4.2,
                        "max_transient_sone": 8.1,
                        "specific_loudness_sum_sone": 12.34,
                        "specific_loudness_summed_exceedance": 1.1629,
                    }
                },
                export_detail={},
            )
        ]

        synced = widget._sync_left_panel_analysis_details()

        self.assertTrue(synced)
        _condition, detail_values = widget.left_panel.analysis_details[-1]
        self.assertEqual(
            detail_values["响度"],
            "稳态平均响度：4.20 sone；特征响度超限总量：116.29 cSones；判定：NG",
        )
        self.assertNotIn("最大瞬态响度", detail_values["响度"])
        self.assertNotIn("特征响度总贡献", detail_values["响度"])

    def test_left_panel_analysis_details_marks_fba_fft_without_threshold(self):
        widget = _DummyManualCycleWidget()
        widget._active_product_condition_key = "q6000"
        widget.analysis_config = {
            "fba": {"type": "FBA", "limit_checked": False},
            "fft": {"type": "FFT", "limit_checked": False},
        }
        widget.data_struct = SimpleNamespace(analysis_result_dict={})
        widget.analysis_window = [
            SimpleNamespace(_sequence_analysis_key="fba", title_name="FBA--通道1"),
            SimpleNamespace(_sequence_analysis_key="fft", title_name="FFT--通道1"),
        ]

        synced = widget._sync_left_panel_analysis_details()

        self.assertTrue(synced)
        _condition, detail_values = widget.left_panel.analysis_details[-1]
        self.assertEqual(detail_values["FBA"], "未启用阈值")
        self.assertEqual(detail_values["FFT"], "未启用阈值")

    def test_recording_completion_marks_condition_and_round_complete(self):
        widget = _DummyManualCycleWidget()

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("q6000", "待判定", "pending"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("待判定", "pending"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("q7000", "待判定", "pending"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("待判定", "pending"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("q8000", "待判定", "pending"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("待判定", "pending"))
        self.assertEqual(widget.left_panel.stages[-1], ("本轮采集完成", "pending"))

    def test_recording_completion_does_not_replace_active_analysis_stage(self):
        widget = _DummyManualCycleWidget()
        widget.on_clicked_player_btn()
        widget.left_panel.set_current_stage("分析中", tone="running")
        widget._mark_manual_product_condition_recording_completed()

        self.assertEqual(widget.left_panel.stages[-1], ("分析中", "running"))

    def test_recording_completion_does_not_expose_legacy_unlabeled_state(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.recorded_signal_info = {"labels": "not_labeled"}

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()

        self.assertIn(("q6000", "待判定", "pending"), widget.left_panel.condition_results)
        self.assertNotIn(("q6000", "完成", "ok"), widget.left_panel.condition_results)

    def test_analysis_completion_uses_channel_overall_judgement(self):
        cases = [
            (["OK", "OK"], ("OK", "ok")),
            (["OK", "NG"], ("NG", "ng")),
            (["NG", "待判定"], ("NG", "ng")),
            (["OK", "待判定"], ("未判定", "pending")),
            (["待判定", "待判定"], ("未判定", "pending")),
        ]
        for channel_verdicts, expected in cases:
            with self.subTest(channel_verdicts=channel_verdicts):
                widget = _DummyManualCycleWidget()
                widget._active_product_condition_key = "q6000"
                widget._build_left_panel_channel_results = lambda: [
                    {"raw_channel": index, "result": verdict}
                    for index, verdict in enumerate(channel_verdicts)
                ]

                widget._finish_product_condition_analysis_display()

                self.assertEqual(
                    widget.left_panel.condition_results[-1],
                    ("q6000", *expected),
                )
                self.assertNotEqual(
                    widget.left_panel.condition_results[-1][1],
                    "未标记",
                )

    def test_last_automatic_gear_updates_round_result_and_stage(self):
        widget = _DummyManualCycleWidget()
        widget._active_product_condition_key = "q8000"
        widget.left_panel.automatic_round_result = ("OK", "ok", True)
        widget._build_left_panel_channel_results = lambda: [
            {"raw_channel": 0, "result": "OK"},
            {"raw_channel": 1, "result": "OK"},
        ]

        widget._finish_product_condition_analysis_display()

        self.assertEqual(widget.left_panel.final_results[-1], ("OK", "ok"))
        self.assertEqual(widget.left_panel.stages[-1], ("本轮完成", "ok"))

    def test_round_completion_does_not_restore_legacy_pending_result(self):
        widget = _DummyManualCycleWidget()
        widget.left_panel.automatic_round_result = ("OK", "ok", True)

        for _condition in widget.product_test_condition_configs:
            widget.on_clicked_player_btn()
            widget._mark_manual_product_condition_recording_completed()
            widget._advance_manual_product_condition_cycle_after_recording()

        self.assertEqual(widget.left_panel.final_results[-1], ("OK", "ok"))
        self.assertEqual(widget.left_panel.stages[-1], ("本轮完成", "ok"))

    def test_left_final_result_uses_current_group_summary_only(self):
        widget = _DummyManualCycleWidget()
        widget._manual_product_condition_group_id = ""
        widget._displayed_manual_product_condition_group_id = "current_group"
        widget.recent_test_session_by_id = {
            "old_1": {
                "session_id": "old_1",
                "group_id": "old_group",
                "condition_key": "q6000",
                "recorded_signal_info": {"labels": "OK"},
            },
            "old_2": {
                "session_id": "old_2",
                "group_id": "old_group",
                "condition_key": "q7000",
                "recorded_signal_info": {"labels": "OK"},
            },
            "old_3": {
                "session_id": "old_3",
                "group_id": "old_group",
                "condition_key": "q8000",
                "recorded_signal_info": {"labels": "OK"},
            },
            "current_1": {
                "session_id": "current_1",
                "group_id": "current_group",
                "condition_key": "q6000",
                "recorded_signal_info": {"labels": "OK"},
            },
            "current_2": {
                "session_id": "current_2",
                "group_id": "current_group",
                "condition_key": "q7000",
                "recorded_signal_info": {"labels": "NG"},
            },
            "current_3": {
                "session_id": "current_3",
                "group_id": "current_group",
                "condition_key": "q8000",
                "recorded_signal_info": {"labels": "OK"},
            },
        }

        self.assertIsNone(widget._refresh_current_manual_product_final_from_group("old_group"))
        self.assertEqual(widget.left_panel.final_results, [])

        self.assertEqual(widget._refresh_current_manual_product_final_from_group("current_group"), "NG")
        self.assertEqual(widget.left_panel.final_results[-1], ("NG", "ng"))

    def test_left_final_result_uses_current_recent_session_group_when_display_group_missing(self):
        widget = _DummyManualCycleWidget()
        widget._manual_product_condition_group_id = ""
        widget._displayed_manual_product_condition_group_id = ""
        widget._current_recent_session_id = "current_3"
        widget.recent_test_session_by_id = {
            "current_1": {
                "session_id": "current_1",
                "group_id": "current_group",
                "condition_key": "q6000",
                "recorded_signal_info": {"labels": "OK"},
            },
            "current_2": {
                "session_id": "current_2",
                "group_id": "current_group",
                "condition_key": "q7000",
                "recorded_signal_info": {"labels": "NG"},
            },
            "current_3": {
                "session_id": "current_3",
                "group_id": "current_group",
                "condition_key": "q8000",
                "recorded_signal_info": {"labels": "not_labeled"},
            },
        }

        self.assertEqual(widget._refresh_current_manual_product_final_from_group("current_group"), "NG")
        self.assertEqual(widget.left_panel.final_results[-1], ("NG", "ng"))

    def test_incomplete_round_detects_mid_manual_product_cycle(self):
        widget = _DummyManualCycleWidget()

        self.assertFalse(widget._has_incomplete_manual_product_condition_round())

        widget.on_clicked_player_btn()
        self.assertTrue(widget._has_incomplete_manual_product_condition_round())

        widget._mark_manual_product_condition_recording_completed()
        widget._advance_manual_product_condition_cycle_after_recording()
        self.assertTrue(widget._has_incomplete_manual_product_condition_round())

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        widget._advance_manual_product_condition_cycle_after_recording()
        self.assertTrue(widget._has_incomplete_manual_product_condition_round())

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        widget._advance_manual_product_condition_cycle_after_recording()
        self.assertFalse(widget._has_incomplete_manual_product_condition_round())

    def test_history_partial_group_counts_as_incomplete_round(self):
        widget = _DummyManualCycleWidget()
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": "group_1",
                "condition_key": "q6000",
                "result_label": "not labeled",
                "recorded_signal_info": {"labels": "not_labeled"},
            }
        }

        self.assertTrue(widget._has_incomplete_manual_product_condition_round())

        widget.recent_test_session_by_id.update(
            {
                "recent_2": {
                    "session_id": "recent_2",
                    "group_id": "group_1",
                    "condition_key": "q7000",
                    "result_label": "not labeled",
                    "recorded_signal_info": {"labels": "not_labeled"},
                },
                "recent_3": {
                    "session_id": "recent_3",
                    "group_id": "group_1",
                    "condition_key": "q8000",
                    "result_label": "not labeled",
                    "recorded_signal_info": {"labels": "not_labeled"},
                },
            }
        )
        self.assertFalse(widget._has_incomplete_manual_product_condition_round())

    def test_reset_manual_product_cycle_refreshes_left_panel_to_pending(self):
        widget = _DummyManualCycleWidget()

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()

        widget._reset_manual_product_condition_cycle(clear_waveforms=True)

        self.assertEqual(widget._manual_product_condition_index, 0)
        self.assertEqual(widget._manual_product_condition_group_id, "")
        self.assertEqual(widget._active_product_condition_key, "")
        self.assertEqual(
            widget.left_panel.condition_results[-3:],
            [
                ("q6000", "待检测", "pending"),
                ("q7000", "待检测", "pending"),
                ("q8000", "待检测", "pending"),
            ],
        )
        self.assertEqual(widget.left_panel.final_results[-1], ("待判定", "pending"))
        self.assertEqual(widget.cleared_waveforms, 2)

    def test_manual_product_mark_result_counts_once_after_full_group(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": "group_1",
                "condition_key": "q6000",
                "result_label": "ok",
                "recorded_signal_info": {"labels": "OK"},
            },
            "recent_2": {
                "session_id": "recent_2",
                "group_id": "group_1",
                "condition_key": "q7000",
                "result_label": "ng",
                "recorded_signal_info": {"labels": "NG"},
            },
            "recent_3": {
                "session_id": "recent_3",
                "group_id": "group_1",
                "condition_key": "q8000",
                "result_label": "not labeled",
                "recorded_signal_info": {"labels": "not_labeled"},
            },
        }

        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_results, ["NG"])

        widget.recent_test_session_by_id["recent_3"]["result_label"] = "ok"
        widget.recent_test_session_by_id["recent_3"]["recorded_signal_info"]["labels"] = "OK"
        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_3"))
        self.assertEqual(widget.count_board.mark_results, ["NG"])
        self.assertEqual(widget.count_board.mark_relabels, [])
        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_3"))
        self.assertEqual(widget.count_board.mark_results, ["NG"])

    def test_single_condition_mark_result_uses_product_group_summary(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.product_test_condition_configs = [
            {"key": "q6000", "condition_name": "6000", "test_queue": "queue_6000"},
        ]
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": "group_1",
                "condition_key": "q6000",
                "result_label": "not labeled",
                "recorded_signal_info": {"labels": "not_labeled"},
            },
        }

        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_results, ["not_labeled"])

        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_results, ["not_labeled"])

        widget.recent_test_session_by_id["recent_1"]["result_label"] = "ng"
        widget.recent_test_session_by_id["recent_1"]["recorded_signal_info"]["labels"] = "NG"
        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_relabels, [("not_labeled", "NG")])

    def test_manual_product_group_count_rolls_back_when_label_returns_to_not_labeled(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": "group_1",
                "condition_key": "q6000",
                "result_label": "ok",
                "recorded_signal_info": {"labels": "OK"},
            },
            "recent_2": {
                "session_id": "recent_2",
                "group_id": "group_1",
                "condition_key": "q7000",
                "result_label": "ok",
                "recorded_signal_info": {"labels": "OK"},
            },
            "recent_3": {
                "session_id": "recent_3",
                "group_id": "group_1",
                "condition_key": "q8000",
                "result_label": "ok",
                "recorded_signal_info": {"labels": "OK"},
            },
        }

        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_results, ["OK"])

        widget.recent_test_session_by_id["recent_1"]["result_label"] = "not labeled"
        widget.recent_test_session_by_id["recent_1"]["recorded_signal_info"]["labels"] = "not_labeled"
        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_relabels[-1], ("OK", "not_labeled"))

        widget.recent_test_session_by_id["recent_1"]["result_label"] = "ok"
        widget.recent_test_session_by_id["recent_1"]["recorded_signal_info"]["labels"] = "OK"
        self.assertTrue(widget._update_manual_product_mark_group_count_for_session("recent_1"))
        self.assertEqual(widget.count_board.mark_relabels[-1], ("not_labeled", "OK"))


class _CanonicalHistoryHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self._recent_session_max_items = 20
        self._condition_record_cache = {}
        self._manual_product_condition_group_id = "active"
        self._manual_product_condition_results = {"01": "NG", "03": "NG"}
        self._manual_product_condition_completed_keys = {"03", "04"}

    def _product_condition_sequence(self):
        return [{"key": key} for key in ("01", "02", "03", "04")]

    def _product_condition_runtime_key(self, condition, index=0):
        return condition["key"]

    def _build_recent_session_record(self, result_label):
        return dict(self.next_record, result_label=result_label)


def test_canonical_history_aggregation_precedence_and_metadata_without_panel():
    host = _CanonicalHistoryHost()
    host.recent_test_session_by_id = {
        "old": {"group_id": "active", "condition_key": "01", "result_label": "NG"},
        "latest": {"group_id": "active", "condition_key": "01", "result_label": "OK",
                   "barcode": "SN1", "product_model": "M1", "time_text": "now"},
        "other": {"group_id": "other", "condition_key": "02", "result_label": "NG"},
    }
    host._condition_record_cache = {
        "01": {"group_id": "active", "condition_key": "01", "source_type": "imported", "result_label": "NG"},
        "02": {"group_id": "active", "condition_key": "02", "source_type": "restored", "result_label": "OK"},
        "ignored": {"group_id": "active", "condition_key": "ignored", "source_type": "recorded", "result_label": "NG"},
    }
    group = host._collect_product_condition_records("active")
    assert group["records"]["01"] is host.recent_test_session_by_id["latest"]
    assert group["results"] == {"01": "OK", "02": "OK", "03": "NG", "04": "not_labeled"}
    assert (group["barcode"], group["product_model"], group["time_text"]) == ("SN1", "M1", "now")
    assert host._product_group_result_state("active") == (True, "NG")
    assert host._product_group_result_state("other") == (False, None)
    assert host._collect_product_condition_records("missing") is None
    host._condition_record_cache["02"]["source_type"] = "imported"
    assert host._collect_product_condition_records("active")["results"]["02"] == "OK"


def test_canonical_history_retains_twenty_records_and_evicts_same_or_other_groups():
    for same_group in (False, True):
        host = _CanonicalHistoryHost()
        for index in range(21):
            host.next_record = {"session_id": str(index), "group_id": "kept" if same_group or index else "evicted",
                                "condition_key": "01" if index == 0 else "02"}
            host._append_recent_session_from_current_run("OK")
        assert len(host.recent_test_sessions) == len(host.recent_test_session_by_id) == 20
        assert host.recent_test_sessions == [str(index) for index in range(20, 0, -1)]
        assert "0" not in host.recent_test_session_by_id
        assert host._current_recent_session_id == "20"
        assert host._pending_recent_session_append is False
        assert host._collect_product_condition_records("evicted") is None
        assert host._collect_product_condition_records("kept")["results"] == {"02": "OK"}


class _GroupMetadataHistoryHost(_CanonicalHistoryHost):
    def _product_condition_sequence(self):
        return [{"key": "01", "condition_name": "Low"},
                {"key": "02", "condition_name": "High"}]

    def append_record(self, index, group_id="group", condition_key="01"):
        self.next_record = {
            "session_id": str(index), "group_id": group_id,
            "condition_key": condition_key, "mode": condition_key,
            "created_at": f"2026-10-09T10:{index:02d}:00",
            "time_text": f"2026-10-09 10:{index:02d}:00",
            "barcode": f"SN{index}", "product_model": f"Model{index}",
            "recorded_path": f"D:/audio/{index}.wav", "sample_rate": 48000,
            "recorded_signal_info": {"labels": "OK"},
            "analysis_result_dict": {}, "config_snapshot": {},
        }
        self._append_recent_session_from_current_run("OK")

    def _refresh_manual_product_condition_results_from_group(self, group_id):
        return None

    def _refresh_current_manual_product_final_from_group(self, group_id):
        return None


def test_group_metadata_retest_preserves_first_time_and_latest_upsert():
    host = _GroupMetadataHistoryHost()
    host.append_record(0)
    host.append_record(1, condition_key="02")
    host.append_record(2)

    group = host._collect_product_condition_records("group")
    assert group["time_text"] == "2026-10-09 10:00:00"
    assert (group["barcode"], group["product_model"]) == ("SN2", "Model2")
    assert group["records"]["01"] is host.recent_test_session_by_id["2"]
    assert group["records"]["01"]["time_text"] == "2026-10-09 10:02:00"


def test_group_metadata_older_record_update_uses_upsert_order_without_changing_time():
    host = _GroupMetadataHistoryHost()
    host.append_record(0)
    host.append_record(1, condition_key="02")
    host.append_record(2)
    host._update_recent_session("0", barcode="Corrected", product_model="Revised",
                                time_text="2026-10-09 11:00:00")

    group = host._collect_product_condition_records("group")
    assert (group["barcode"], group["product_model"]) == ("Corrected", "Revised")
    assert group["time_text"] == "2026-10-09 10:00:00"
    assert group["records"]["01"] is host.recent_test_session_by_id["2"]
    assert host.recent_test_session_by_id["0"]["time_text"] == "2026-10-09 11:00:00"
    host._update_recent_session("1", barcode="", product_model="")
    group = host._collect_product_condition_records("group")
    assert (group["barcode"], group["product_model"]) == ("Corrected", "Revised")


def test_group_metadata_survives_same_and_cross_group_eviction_then_expires():
    for same_group in (True, False):
        host = _GroupMetadataHistoryHost()
        for index in range(21):
            group_id = "group" if same_group or index < 2 else "other"
            host.append_record(index, group_id, "01" if index % 2 == 0 else "02")

        group = host._collect_product_condition_records("group")
        assert len(host.recent_test_session_by_id) == len(host.recent_test_sessions) == 20
        assert "0" not in host.recent_test_session_by_id
        assert group["time_text"] == "2026-10-09 10:00:00"
        latest = 20 if same_group else 1
        assert (group["barcode"], group["product_model"]) == (f"SN{latest}", f"Model{latest}")
        assert host.recent_test_session_by_id["1"]["time_text"] == "2026-10-09 10:01:00"
        if not same_group:
            host.append_record(21, "other")
            assert host._collect_product_condition_records("group") is None
            host.append_record(22)
            group = host._collect_product_condition_records("group")
            assert group["time_text"] == "2026-10-09 10:22:00"
            assert (group["barcode"], group["product_model"]) == ("SN22", "Model22")


if __name__ == "__main__":
    unittest.main()
