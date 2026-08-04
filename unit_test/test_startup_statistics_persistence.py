import json
import logging
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _SpyCountBoard:
    def __init__(self):
        self.test_refresh_count = 0
        self.mark_refresh_count = 0

    def set_test_text(self):
        self.test_refresh_count += 1

    def set_mark_text(self):
        self.mark_refresh_count += 1


class _SpyLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(str(message))


class _DummyStartupStatsWidget(SequenceWidgetStreamingOpsMixin):
    def __init__(self):
        self.analysis_config = {}
        self.count_board = _SpyCountBoard()
        self.default_logger = _SpyLogger()


class TestStartupStatisticsPersistence(unittest.TestCase):
    def _write_test_result(self, folder, date_text, total, ok, ng, not_labels=0):
        os.makedirs(os.path.join(folder, "log", "test_result_log"), exist_ok=True)
        ok_percent = round(ok / total * 100, 2) if total > 0 else 0
        with open(os.path.join(folder, "log", "test_result_log", f"{date_text}.dat"), "w") as f:
            f.writelines(
                [
                    f"total: {total}\n",
                    f"ok: {ok}\n",
                    f"ng: {ng}\n",
                    f"not_labels: {not_labels}\n",
                    f"ok_percent: {ok_percent}%\n",
                    f"datatime: {date_text}\n",
                ]
            )

    def _write_mark_result(self, folder, date_text, total, ok, ng, not_labels=0):
        os.makedirs(os.path.join(folder, "ui", "ui_config"), exist_ok=True)
        with open(os.path.join(folder, "ui", "ui_config", "mark_result.json"), "w") as f:
            json.dump(
                {
                    "total": total,
                    "ok": ok,
                    "ng": ng,
                    "not_labels": not_labels,
                    "datatime": date_text,
                },
                f,
            )

    def _read_test_result(self, folder, date_text):
        with open(os.path.join(folder, "log", "test_result_log", f"{date_text}.dat")) as f:
            return f.read()

    def _read_mark_result(self, folder):
        with open(os.path.join(folder, "ui", "ui_config", "mark_result.json")) as f:
            return json.load(f)

    def test_startup_keeps_same_day_summary_counts(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as folder:
            default_dir = folder + os.sep
            self._write_test_result(folder, current_date, total=6, ok=2, ng=3, not_labels=1)
            self._write_mark_result(folder, current_date, total=8, ok=5, ng=2, not_labels=1)

            with patch("ui.sequence.sequence_widget_streaming_ops.DEFAULT_DIR", default_dir), patch(
                "base.save_data.DEFAULT_DIR",
                default_dir,
            ):
                widget = _DummyStartupStatsWidget()
                widget.reset_statistics_on_startup()

            self.assertIn("total: 6", self._read_test_result(folder, current_date))
            self.assertIn("ok: 2", self._read_test_result(folder, current_date))
            self.assertIn("ng: 3", self._read_test_result(folder, current_date))
            self.assertIn("not_labels: 1", self._read_test_result(folder, current_date))
            self.assertEqual(
                self._read_mark_result(folder),
                {
                    "total": 8,
                    "ok": 5,
                    "ng": 2,
                    "not_labels": 1,
                    "datatime": current_date,
                },
            )
            self.assertEqual(widget.count_board.test_refresh_count, 1)
            self.assertEqual(widget.count_board.mark_refresh_count, 1)

    def test_startup_rolls_mark_summary_to_zero_on_new_day(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as folder:
            default_dir = folder + os.sep
            self._write_mark_result(folder, previous_date, total=8, ok=5, ng=2, not_labels=1)

            with patch("ui.sequence.sequence_widget_streaming_ops.DEFAULT_DIR", default_dir), patch(
                "base.save_data.DEFAULT_DIR",
                default_dir,
            ):
                widget = _DummyStartupStatsWidget()
                widget.reset_statistics_on_startup()

            self.assertIn("total: 0", self._read_test_result(folder, current_date))
            self.assertIn("not_labels: 0", self._read_test_result(folder, current_date))
            self.assertEqual(
                self._read_mark_result(folder),
                {
                    "total": 0,
                    "ok": 0,
                    "ng": 0,
                    "not_labels": 0,
                    "datatime": current_date,
                },
            )


if __name__ == "__main__":
    unittest.main()
