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
    def __init__(self):
        self.condition_results = []
        self.final_results = []
        self.stages = []
        self.analysis_details = []

    def set_condition_result(self, condition, result_text, tone=None):
        self.condition_results.append((condition, result_text, tone))

    def set_final_result(self, result_text, tone=None):
        self.final_results.append((result_text, tone))

    def set_current_stage(self, stage_text, tone=None):
        self.stages.append((stage_text, tone))

    def set_condition_analysis_details(self, condition, detail_values):
        self.analysis_details.append((condition, dict(detail_values or {})))
        return True


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
        self.left_panel = _SpyLeftPanel()
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
        self.loaded_queues.append(condition_config["test_queue"])
        self.sequence_config = [{"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 44100}}, "analysis_list": {}}}]
        return True, ""

    def _generate_recording_token(self):
        self._token_seq += 1
        return f"token_{self._token_seq}"

    def clear_all_direction_waveforms(self):
        self.cleared_waveforms += 1

    def start_this_play(self, label="not_labeled"):
        self.started.append((label, self._active_product_condition_key, self._resolve_recording_name_suffix()))


class TestManualProductConditionCycle(unittest.TestCase):
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

    def test_manual_product_condition_runtime_key_prefers_trigger_state(self):
        widget = _DummyManualCycleWidget()
        widget.product_test_condition_configs = [
            {"key": "uuid_6000", "trigger_state": "01", "condition_name": "6000", "test_queue": "queue_6000"},
            {"key": "uuid_7000", "trigger_state": "02", "condition_name": "7000", "test_queue": "queue_7000"},
            {"key": "uuid_8000", "trigger_state": "03", "condition_name": "8000", "test_queue": "queue_8000"},
        ]

        widget.on_clicked_player_btn()
        self.assertEqual(widget.started[-1], ("not_labeled", "01", "_6000"))
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("01", "完成", "ok"), widget.left_panel.condition_results)

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget.started[-1], ("not_labeled", "02", "_7000"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        self.assertEqual(widget.started[-1], ("not_labeled", "03", "_8000"))

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

    def test_left_panel_analysis_details_syncs_runtime_metrics(self):
        widget = _DummyManualCycleWidget()
        widget._active_product_condition_key = "q6000"
        widget.analysis_config = {
            "spl": {"type": "SPL", "weighting": "Z"},
            "loudness": {"type": "LOUD", "advanced": {"curve_y_unit": "sone"}},
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

        synced = widget._sync_left_panel_analysis_details(
            {
                "has_ai_analysis": True,
                "label": "OK",
                "scores": {"ok_score": 71.6, "ng_score": 28.4},
            }
        )

        self.assertTrue(synced)
        condition, detail_values = widget.left_panel.analysis_details[-1]
        self.assertEqual(condition, "q6000")
        self.assertEqual(detail_values["SPL"], "总体声压：72.34 dB；判定：OK")
        self.assertEqual(detail_values["响度"], "稳态平均响度：4.20 sone；最大瞬态响度：8.10 sone；判定：OK")
        self.assertEqual(detail_values["AI分析"], "OK Score：71.60%；NG Score：28.40%；判定：OK")
        self.assertEqual(detail_values["FBA"], "OK")
        self.assertEqual(detail_values["FFT"], "NG")

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
        self.assertIn(("q6000", "完成", "ok"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("检测中", "running"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("q7000", "完成", "ok"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("检测中", "running"))

        widget._advance_manual_product_condition_cycle_after_recording()
        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()
        self.assertIn(("q8000", "完成", "ok"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("未标记", "pending"))
        self.assertEqual(widget.left_panel.stages[-1], ("本轮采集完成", "ok"))

    def test_mark_mode_recording_completion_shows_unlabeled_instead_of_completed(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.recorded_signal_info = {"labels": "not_labeled"}

        widget.on_clicked_player_btn()
        widget._mark_manual_product_condition_recording_completed()

        self.assertIn(("q6000", "未标记", "pending"), widget.left_panel.condition_results)
        self.assertNotIn(("q6000", "完成", "ok"), widget.left_panel.condition_results)

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
        widget.recent_session_panel = None
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

    def test_manual_product_group_count_rolls_back_when_label_returns_to_not_labeled(self):
        widget = _DummyManualCycleWidget()
        widget.count_board.mode = "mark"
        widget.recent_session_panel = None
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


if __name__ == "__main__":
    unittest.main()
