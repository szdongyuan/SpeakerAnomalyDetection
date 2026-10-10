"""Retired automatic export must not participate in recording or exit."""
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow

from main_window import MainWindow
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


def test_canonical_record_keeps_results_without_automatic_pdf_fields():
    class Host(SequenceWidgetAnalysisOpsMixin):
        def _get_recent_session_mode_text(self, direction):
            return "Low"

        def _get_recent_session_mode_key(self, direction):
            return "01"

    host = Host()
    host.recorded_path = "record.wav"
    host.recorded_signal_info = {"labels": "OK", "sample_number": 7}
    host._recent_session_seq = 0
    host.lineedit_s_or_n = SimpleNamespace(text=lambda: "SN1")
    host.lineedit_type = SimpleNamespace(text=lambda: "Model1")
    host.data_struct = SimpleNamespace(sample_rate=48000, analysis_result_dict={"SPL": (True, 0.2)})
    record = host._build_recent_session_record("OK")
    assert record["analysis_result_dict"] == {"SPL": (True, 0.2)}
    assert record["recorded_signal_info"] == host.recorded_signal_info
    assert (record["barcode"], record["product_model"], record["condition_key"]) == ("SN1", "Model1", "01")
    assert record["created_at"] and record["time_text"]
    assert "analysis_report_state" not in record
    assert "analysis_report_items" not in record


def test_main_exit_closes_sequence_and_progress_without_pdf_callback(ui_qapp):
    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.sequence_window = SimpleNamespace(
                _analysis_has_pending_tasks=lambda: False,
                _begin_raw_audio_csv_close=Mock(),
                _save_product_test_progress_before_exit=Mock(),
                close=Mock(),
            )
            self._close_all_subwindows = Mock()

    window = Window()
    # A stale dynamically supplied callback must no longer be discovered.
    retired_callback = Mock()
    window.sequence_window._shutdown_product_pdf_exporter = retired_callback
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    window.sequence_window._begin_raw_audio_csv_close.assert_called_once_with(application_exit=True)
    window.sequence_window._save_product_test_progress_before_exit.assert_called_once_with()
    window.sequence_window.close.assert_called_once_with()
    window._close_all_subwindows.assert_called_once_with()
    retired_callback.assert_not_called()
    window.deleteLater()
