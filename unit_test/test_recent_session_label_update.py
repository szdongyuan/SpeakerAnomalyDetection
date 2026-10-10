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

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class _LineEdit:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text


class _DummySequenceWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.count_board = SimpleNamespace(mode="mark")
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

    def test_current_record_preserves_frozen_configuration_and_product_metadata(self):
        widget = _DummySequenceWidget()
        widget.sequence_config = [{"seq1": {"acq": {"detail": {"sample_rate": 48000}}}}]
        widget.analysis_config = {"display_sequence": ["spl"], "spl": {"limit": 3}}
        widget._active_product_condition_config = {"key": "01"}
        widget._active_input_channels = [2, 7]
        widget.using_config_path = "queue.json"

        record = widget._build_recent_session_record("OK")
        widget.sequence_config[0]["seq1"]["acq"]["detail"]["sample_rate"] = 44100
        widget.analysis_config["spl"]["limit"] = 9
        widget._active_input_channels.append(8)

        snapshot = record["config_snapshot"]
        self.assertEqual(snapshot["sequence_config"][0]["seq1"]["acq"]["detail"]["sample_rate"], 48000)
        self.assertEqual(snapshot["analysis_config"]["spl"]["limit"], 3)
        self.assertEqual(snapshot["active_input_channels"], [2, 7])
        self.assertEqual(snapshot["condition_config"], {"key": "01"})
        self.assertEqual(snapshot["using_config_path"], "queue.json")
        self.assertEqual((record["barcode"], record["product_model"]), ("SN001", "MODEL"))

    def test_recent_session_group_id_prefers_cycle_token(self):
        widget = _DummySequenceWidget()
        widget._current_cycle_recorded_count = "cycle_1"
        widget._current_run_recording_token = "run_a"

        session_record = widget._build_recent_session_record("OK")

        self.assertEqual(session_record["group_id"], "cycle_1")


if __name__ == "__main__":
    unittest.main()
