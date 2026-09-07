import logging
import sys
import types
import unittest
from types import SimpleNamespace

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

        def emit(self, record):
            return None

        def close(self):
            super().close()

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from consts import error_code
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class _LineEdit:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text


class _DummyRecentSessionPanel:
    def __init__(self):
        self.upserted_records = []

    def upsert_session(self, session_record):
        self.upserted_records.append(dict(session_record))


class _SpyLeftPanel:
    def __init__(self):
        self.condition_results = []
        self.final_results = []

    def set_condition_result(self, condition, result_text, tone=None):
        self.condition_results.append((condition, result_text, tone))

    def set_final_result(self, result_text, tone=None):
        self.final_results.append((result_text, tone))


class _DummySequenceWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.count_board = SimpleNamespace(mode="mark")
        self.count_board_relabel_updates = []
        self.count_board.update_mark_result_file_on_relabel = self._record_mark_result_update
        self.recent_session_panel = _DummyRecentSessionPanel()
        self.recent_test_sessions = []
        self._recent_session_seq = 0
        self._recent_session_max_items = 20
        self._current_recent_session_id = None
        self._pending_recent_session_append = False
        self._current_trigger_direction = "01"
        self._current_cycle_recorded_count = ""
        self._current_run_recording_token = "run_001"
        self.lineedit_s_or_n = _LineEdit("SN001")
        self.lineedit_type = _LineEdit("MODEL")
        self.product_test_condition_configs = [
            {"key": "01", "trigger_state": "01", "condition_name": "6000 rpm"},
            {"key": "02", "trigger_state": "02", "condition_name": "7000 rpm"},
        ]
        self.data_struct = SimpleNamespace(sample_rate=44100, analysis_result_dict={})
        self.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "result_label": "not labeled",
                "recorded_path": "D:/audio_data/stored_data/not_labeled/test.wav",
                "recorded_signal_info": {
                    "file_path": "audio_data/stored_data/not_labeled/test.wav",
                    "labels": "not_labeled",
                },
            }
        }
        self.recorded_path = "D:/audio_data/stored_data/not_labeled/test.wav"
        self.recorded_signal_info = {
            "file_path": "audio_data/stored_data/not_labeled/test.wav",
            "labels": "not_labeled",
        }
        self.relabel_calls = []

    def _record_mark_result_update(self, old_label, new_label):
        self.count_board_relabel_updates.append((old_label, new_label))

    def _resolve_recent_session_path(self, session_record):
        return session_record.get("recorded_path")

    def _relabel_stored_audio_record(self, recorded_path, recorded_signal_info, label):
        self.relabel_calls.append((recorded_path, dict(recorded_signal_info), label))
        folder = label if label in ("OK", "NG") else "not_labeled"
        return (
            error_code.OK,
            "ok",
            f"D:/audio_data/stored_data/{folder}/test.wav",
            {
                "file_path": f"audio_data/stored_data/{folder}/test.wav",
                "labels": label,
            },
        )


class TestRecentSessionLabelUpdate(unittest.TestCase):
    def test_recent_session_mode_text_uses_condition_names(self):
        widget = _DummySequenceWidget()
        widget.product_test_condition_configs = [
            {"key": "01", "trigger_state": "01", "condition_name": "6000 rpm"},
            {"key": "02", "trigger_state": "02", "condition_name": "7000 rpm"},
        ]

        self.assertEqual(widget._get_recent_session_mode_text("01"), "6000 rpm")
        self.assertEqual(widget._get_recent_session_mode_text("02"), "7000 rpm")
        self.assertEqual(widget._get_recent_session_mode_text("forward"), "6000 rpm")
        self.assertEqual(widget._get_recent_session_mode_text("reverse"), "7000 rpm")
        self.assertEqual(widget._get_recent_session_mode_key("reverse"), "02")

    def test_change_recent_session_result_updates_session_and_runtime_paths(self):
        widget = _DummySequenceWidget()

        changed = widget._change_recent_session_result_by_id("recent_1", "OK")

        self.assertTrue(changed)
        self.assertEqual(
            widget.relabel_calls,
            [
                (
                    "D:/audio_data/stored_data/not_labeled/test.wav",
                    {
                        "file_path": "audio_data/stored_data/not_labeled/test.wav",
                        "labels": "not_labeled",
                    },
                    "OK",
                )
            ],
        )
        self.assertEqual(widget.recent_test_session_by_id["recent_1"]["result_label"], "ok")
        self.assertEqual(widget.recent_test_session_by_id["recent_1"]["recorded_path"], "D:/audio_data/stored_data/OK/test.wav")
        self.assertEqual(
            widget.recent_test_session_by_id["recent_1"]["recorded_signal_info"],
            {
                "file_path": "audio_data/stored_data/OK/test.wav",
                "labels": "OK",
            },
        )
        self.assertEqual(widget.recorded_path, "D:/audio_data/stored_data/OK/test.wav")
        self.assertEqual(
            widget.recorded_signal_info,
            {
                "file_path": "audio_data/stored_data/OK/test.wav",
                "labels": "OK",
            },
        )
        self.assertEqual(
            widget.recent_session_panel.upserted_records[-1]["recorded_path"],
            "D:/audio_data/stored_data/OK/test.wav",
        )
        self.assertEqual(widget.count_board_relabel_updates, [("not_labeled", "OK")])

    def test_change_recent_session_result_syncs_matching_waveform_record(self):
        widget = _DummySequenceWidget()
        widget.recent_test_session_by_id["recent_1"].update(
            {
                "condition_key": "01",
                "result_label": "ok",
                "recorded_path": "D:/audio_data/stored_data/OK/test.wav",
                "recorded_signal_info": {
                    "file_path": "audio_data/stored_data/OK/test.wav",
                    "labels": "OK",
                },
            }
        )
        widget.recorded_path = "D:/audio_data/stored_data/OK/test.wav"
        widget.recorded_signal_info = {
            "file_path": "audio_data/stored_data/OK/test.wav",
            "labels": "OK",
        }
        widget._condition_record_cache = {
            "01": {
                "recorded_path": "D:/audio_data/stored_data/OK/test.wav",
                "recorded_signal_info": {
                    "file_path": "audio_data/stored_data/OK/test.wav",
                    "labels": "OK",
                },
                "session_id": "recent_1",
            }
        }
        result_updates = []
        path_updates = []
        widget.channel_workspace = SimpleNamespace(
            set_condition_result=lambda key, label: result_updates.append((key, label)),
            set_condition_audio_path=lambda key, path: path_updates.append((key, path)),
        )

        changed = widget._change_recent_session_result_by_id("recent_1", "not_labeled")

        self.assertTrue(changed)
        cached = widget._condition_record_cache["01"]
        self.assertEqual(cached["recorded_path"], "D:/audio_data/stored_data/not_labeled/test.wav")
        self.assertEqual(cached["recorded_signal_info"]["labels"], "not_labeled")
        self.assertEqual(cached["session_id"], "recent_1")
        self.assertEqual(result_updates, [("01", "not_labeled")])
        self.assertEqual(
            path_updates,
            [("01", "D:/audio_data/stored_data/not_labeled/test.wav")],
        )

    def test_change_old_session_result_does_not_sync_current_waveform_record(self):
        widget = _DummySequenceWidget()
        widget.recent_test_session_by_id["recent_1"]["condition_key"] = "01"
        widget._condition_record_cache = {
            "01": {
                "recorded_path": "D:/audio_data/current/not_labeled/test.wav",
                "recorded_signal_info": {"labels": "not_labeled"},
                "session_id": "recent_2",
            }
        }
        result_updates = []
        path_updates = []
        widget.channel_workspace = SimpleNamespace(
            set_condition_result=lambda key, label: result_updates.append((key, label)),
            set_condition_audio_path=lambda key, path: path_updates.append((key, path)),
        )

        changed = widget._change_recent_session_result_by_id("recent_1", "OK")

        self.assertTrue(changed)
        self.assertEqual(widget._condition_record_cache["01"]["session_id"], "recent_2")
        self.assertEqual(widget._condition_record_cache["01"]["recorded_signal_info"]["labels"], "not_labeled")
        self.assertEqual(result_updates, [])
        self.assertEqual(path_updates, [])

    def test_change_recent_session_result_refreshes_left_product_condition_state(self):
        widget = _DummySequenceWidget()
        widget.left_panel = _SpyLeftPanel()
        widget._manual_product_condition_group_id = ""
        widget._displayed_manual_product_condition_group_id = ""
        widget._manual_product_condition_results = {}
        widget._manual_product_condition_completed_keys = set()
        widget.recent_test_session_by_id = {
            "recent_1": {
                "session_id": "recent_1",
                "group_id": "group_1",
                "condition_key": "01",
                "result_label": "not labeled",
                "recorded_path": "D:/audio_data/stored_data/not_labeled/test.wav",
                "recorded_signal_info": {
                    "file_path": "audio_data/stored_data/not_labeled/test.wav",
                    "labels": "not_labeled",
                },
            },
            "recent_2": {
                "session_id": "recent_2",
                "group_id": "group_1",
                "condition_key": "02",
                "result_label": "ok",
                "recorded_path": "D:/audio_data/stored_data/OK/test_2.wav",
                "recorded_signal_info": {
                    "file_path": "audio_data/stored_data/OK/test_2.wav",
                    "labels": "OK",
                },
            },
        }

        changed = widget._change_recent_session_result_by_id("recent_1", "OK")

        self.assertTrue(changed)
        self.assertIn(("01", "OK", "ok"), widget.left_panel.condition_results)
        self.assertIn(("02", "OK", "ok"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("OK", "ok"))

        changed = widget._change_recent_session_result_by_id("recent_1", "not_labeled")

        self.assertTrue(changed)
        self.assertIn(("01", "待判定", "pending"), widget.left_panel.condition_results)
        self.assertEqual(widget.left_panel.final_results[-1], ("待判定", "pending"))

    def test_recent_session_group_id_uses_current_run_token(self):
        widget = _DummySequenceWidget()
        widget._current_cycle_recorded_count = ""

        widget._current_run_recording_token = "run_a"
        first = widget._build_recent_session_record("OK")
        widget._current_run_recording_token = "run_b"
        second = widget._build_recent_session_record("OK")

        self.assertEqual(first["group_id"], "run_a")
        self.assertEqual(second["group_id"], "run_b")
        self.assertNotEqual(first["group_id"], second["group_id"])

    def test_recent_session_group_id_prefers_cycle_token(self):
        widget = _DummySequenceWidget()
        widget._current_cycle_recorded_count = "cycle_1"
        widget._current_run_recording_token = "run_a"

        session_record = widget._build_recent_session_record("OK")

        self.assertEqual(session_record["group_id"], "cycle_1")


if __name__ == "__main__":
    unittest.main()
