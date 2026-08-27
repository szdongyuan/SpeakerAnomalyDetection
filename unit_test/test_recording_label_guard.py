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
from consts import error_code


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


class _ButtonSpy:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)


class _SuccessfulGlobalLabelWidget(SequenceWidgetBarcodeOpsMixin):
    def __init__(self):
        self.player_status_flag = False
        self.streaming_processor = SimpleNamespace(is_recording=False)
        self.data_struct = SimpleNamespace(
            store_wave_data=[0.1, 0.2],
            store_wave_data_multi=[[0.1], [0.2]],
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        )
        self.sequence_config = []
        self.recorded_signal_info = {"labels": "not_labeled"}
        self.signal_info = {"old": True}
        self.replayer_btn = _ButtonSpy()
        self.data_btn = _ButtonSpy()
        self.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)

    def update_audio_label_info(self):
        self.recorded_signal_info["labels"] = "OK"

    def _maybe_export_excel_results(self):
        return None

    def update_recorded_signal_info_to_db(self):
        return error_code.OK, "saved"

    def _update_current_recent_session_result(self, _label):
        return None

    def _close_analysis_windows(self):
        return None

    def mark_result(self, previous_label="not_labeled"):
        return None

    def clear_all_direction_waveforms(self):
        return None

    def _reset_barcode_commit_dedup(self):
        return None

    def update_player_btn_is_paused(self):
        return None


class TestRecordingLabelGuard(unittest.TestCase):
    def test_successful_global_label_clears_audio_and_wav_metadata_state(self):
        widget = _SuccessfulGlobalLabelWidget()

        widget.clicked_ok_or_ng()

        self.assertIsNone(widget.data_struct.store_wave_data)
        self.assertIsNone(widget.data_struct.store_wave_data_multi)
        self.assertIsNone(widget.data_struct.wav_calibration_metadata)
        self.assertFalse(widget.data_struct.wav_calibration_metadata_authoritative)
        self.assertFalse(widget.data_struct.wav_calibration_warning_shown)
        self.assertFalse(widget.replayer_btn.enabled)
        self.assertFalse(widget.data_btn.enabled)

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
