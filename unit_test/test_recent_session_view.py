import logging
import sys
import types
import unittest
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QWidget

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


class _DummyButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, value):
        self.enabled = value


class _SelfShowingAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.calculate_count = 0
        self.setWindowTitle("self showing analysis")

    def calculate_spl(self):
        self.calculate_count += 1
        self.show()
        QApplication.processEvents()
        return {"overall_spl": 1.0}


class _SilentRunWidget(QWidget, SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        QWidget.__init__(self)
        self.analysis_instance = _SelfShowingAnalysis()
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self._analysis_window_key_by_obj = {}
        self.analysis_config = {
            "display_sequence": ["声压级 (SPL) 1"],
            "声压级 (SPL) 1": {"type": "SPL"},
        }
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {"mode": "RECORD_ONLY"},
                    "analysis_list": self.analysis_config,
                }
            }
        ]
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self.count_board = SimpleNamespace(mode="view")
        self.recorded_signal_info = {}
        self.recorded_path = ""
        self.persisted_geometry = []

    def _close_analysis_windows(self):
        for window in list(self.analysis_window):
            window.close()
        self.analysis_window = []
        if self._analysis_result_summary_window is not None:
            self._analysis_result_summary_window.close()
            self._analysis_result_summary_window = None

    def instance_analysis_class(self, *_args):
        self.analysis_window.append(self.analysis_instance)

    def _get_analysis_window_geometry(self, _key):
        return None

    def _set_analysis_window_geometry(self, key, geo):
        self.persisted_geometry.append((key, dict(geo)))

    def _can_output_ok_ng(self):
        return False, ""

    def _sync_left_panel_analysis_details(self, _ai_runtime_state=None):
        return False


class TestRecentSessionView(unittest.TestCase):
    def test_silent_run_hides_analysis_windows_even_if_analysis_shows_itself(self):
        app = QApplication.instance() or QApplication([])
        widget = _SilentRunWidget()

        widget.run(show_windows=False)
        app.processEvents()

        self.assertEqual(widget.analysis_instance.calculate_count, 1)
        self.assertFalse(widget.analysis_instance.isVisible())
        widget.analysis_instance.close()
        widget.close()


class _FailingAnalysisWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self._current_recent_session_id = "recent-current"
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self.updated_sessions = []

    def _run_analysis_impl(self, show_windows=True):
        raise RuntimeError("analysis crashed")

    def _update_recent_session(self, session_id, **fields):
        self.updated_sessions.append((session_id, fields))


class _RecentSessionUpdateWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.recent_test_session_by_id = {
            "recent-single": {
                "session_id": "recent-single",
                "group_id": "group-single",
            }
        }
        self.refreshed_groups = []

    def _refresh_manual_product_condition_results_from_group(self, _group_id):
        return None

    def _refresh_current_manual_product_final_from_group(self, group_id):
        self.refreshed_groups.append(group_id)


class TestAnalysisFailureAndSessionUpdate(unittest.TestCase):
    def test_analysis_exception_propagates_without_obsolete_snapshot(self):
        widget = _FailingAnalysisWidget()

        with self.assertRaisesRegex(RuntimeError, "analysis crashed"):
            widget.run(show_windows=False)

        self.assertEqual(widget.updated_sessions, [])

    def test_terminal_session_update_refreshes_group(self):
        widget = _RecentSessionUpdateWidget()

        widget._update_recent_session(
            "recent-single",
            analysis_result_dict={"SPL": (True, 0.0)},
        )

        self.assertEqual(widget.refreshed_groups, ["group-single"])


if __name__ == "__main__":
    unittest.main()
