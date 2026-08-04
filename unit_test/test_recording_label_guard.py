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

from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _DummyCountBoard:
    def __init__(self):
        self.mode = "mark"


class _DummyGlobalLabelWidget(SequenceWidgetBarcodeOpsMixin):
    def __init__(self):
        self.player_status_flag = True
        self.streaming_processor = SimpleNamespace(is_recording=True)
        self.data_struct = SimpleNamespace(store_wave_data=[0.1, 0.2])
        self.sequence_config = []
        self.count_board = _DummyCountBoard()
        self.update_audio_label_calls = 0

    def update_audio_label_info(self):
        self.update_audio_label_calls += 1


class _DummyWaveformLabelWidget(SequenceWidgetStreamingOpsMixin):
    def __init__(self):
        self.player_status_flag = False
        self.streaming_processor = SimpleNamespace(is_recording=True)
        self.count_board = _DummyCountBoard()
        self._condition_record_cache = {}
        self.recorded_path = None
        self.recorded_signal_info = {}
        self._current_recent_session_id = ""
        self.marked_results = []
        self.channel_workspace = SimpleNamespace(
            set_condition_result=lambda key, label: self.marked_results.append((key, label))
        )

    def _resolve_waveform_direction(self, fallback=""):
        return fallback

    def _resolve_active_recording_waveform_direction(self, fallback=""):
        return fallback

    def _waveform_condition_keys(self):
        return ["01"]


class TestRecordingLabelGuard(unittest.TestCase):
    def test_global_ok_ng_is_blocked_while_recording(self):
        widget = _DummyGlobalLabelWidget()

        with patch("ui.sequence.sequence_widget_barcode_ops.QMessageBox.warning") as warning:
            widget.clicked_ok_or_ng()

        warning.assert_called_once()
        self.assertEqual(widget.update_audio_label_calls, 0)

    def test_waveform_ok_ng_is_blocked_while_streaming_processor_records(self):
        widget = _DummyWaveformLabelWidget()

        with patch("ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning") as warning:
            widget.on_waveform_condition_mark_clicked("01", "OK")

        warning.assert_called_once()
        self.assertEqual(widget.marked_results, [])

    def test_waveform_ok_ng_is_blocked_before_condition_recording_completes(self):
        widget = _DummyWaveformLabelWidget()
        widget.streaming_processor = SimpleNamespace(is_recording=False)

        with patch("ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning") as warning:
            widget.on_waveform_condition_mark_clicked("01", "OK")

        warning.assert_called_once()
        self.assertIn("录音尚未完成", warning.call_args[0][2])
        self.assertEqual(widget.marked_results, [])


if __name__ == "__main__":
    unittest.main()
