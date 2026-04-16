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


class _DummyRecentSessionPanel:
    def __init__(self):
        self.upserted_records = []

    def upsert_session(self, session_record):
        self.upserted_records.append(dict(session_record))


class _DummySequenceWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.count_board = SimpleNamespace(mode="mark")
        self.count_board_relabel_updates = []
        self.count_board.update_mark_result_file_on_relabel = self._record_mark_result_update
        self.recent_session_panel = _DummyRecentSessionPanel()
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
        return (
            error_code.OK,
            "ok",
            "D:/audio_data/stored_data/OK/test.wav",
            {
                "file_path": "audio_data/stored_data/OK/test.wav",
                "labels": "OK",
            },
        )


class TestRecentSessionLabelUpdate(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
