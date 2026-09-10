import logging
import sys
import types
import unittest
import copy
from types import SimpleNamespace

import numpy as np
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

    def _capture_excel_export_cache(self):
        return None

    def _maybe_export_excel_results(self):
        return None

    def _can_output_ok_ng(self):
        return False, ""

    def _sync_left_panel_analysis_details(self, _ai_runtime_state=None):
        return False


class _DummySequenceWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self._session_record = {"recorded_signal_info": {}, "sample_rate": 48000}
        self.recorded_path = "current.wav"
        self.recorded_signal_info = {"file_path": "current.wav"}
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000}},
                    "analysis_list": {
                        "display_sequence": ["fft_current"],
                        "fft_current": {"type": "FFT"},
                    },
                }
            }
        ]
        self.using_config_path = "current_queue.json"
        self.data_struct = SimpleNamespace(
            store_wave_data=np.array([1.0, 2.0], dtype=np.float32),
            store_wave_data_multi=np.array([[1.0], [2.0]], dtype=np.float32),
            sample_rate=48000,
            audio_lenth=2,
            analysis_result_dict={"legacy": (True, 0.0)},
            fft_flag=0,
            stft_flag=0,
        )
        self.count_board = SimpleNamespace(mode="test")
        self._excel_export_cache = {"cached": True}
        self._excel_exported_record_id = "current.wav"
        self.analysis_config = self.sequence_config[0]["seq1"]["analysis_list"]
        self.count_board.analysis_config = self.analysis_config
        self._active_input_channels = [0]
        self.data_btn = _DummyButton()
        self.run_calls = []
        self.run_analysis_configs = []
        self.run_sequence_configs = []
        self.closed_windows = 0
        self.loaded_audio = []
        self.plot_calls = []
        self.clear_plot_calls = 0

    def _resolve_recent_session(self, session_id: str):
        return self._session_record if session_id == "recent_1" else None

    def _resolve_recent_session_path(self, session_record):
        return "history.wav"

    def _close_analysis_windows(self):
        self.closed_windows += 1

    def _load_audio_file_to_data_struct(self, file_path: str, sample_rate=None):
        self.loaded_audio.append((file_path, sample_rate))
        self.data_struct.store_wave_data = np.array([3.0, 4.0], dtype=np.float32)
        self.data_struct.store_wave_data_multi = np.array([[3.0], [4.0]], dtype=np.float32)
        self.data_struct.sample_rate = sample_rate or 48000
        self.data_struct.audio_lenth = 2
        self.data_struct.analysis_result_dict = {"viewed": (True, 0.0)}

    def run(
        self,
        show_windows=True,
        *,
        report_session_id=None,
        capture_product_report=True,
    ):
        self.run_calls.append(
            {
                "show_windows": show_windows,
                "report_session_id": report_session_id,
                "capture_product_report": capture_product_report,
            }
        )
        self.run_analysis_configs.append(copy.deepcopy(self.analysis_config))
        self.run_sequence_configs.append(copy.deepcopy(self.sequence_config))

    def plot_waveform_to_workspace(self, data, sample_rate):
        self.plot_calls.append((np.asarray(data).copy(), sample_rate))

    def _clear_plot_area(self):
        self.clear_plot_calls += 1


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

    def test_view_recent_session_does_not_capture_product_report(self):
        widget = _DummySequenceWidget()

        widget._show_recent_session_analysis_by_id("recent_1")

        self.assertEqual(
            widget.run_calls,
            [
                {
                    "show_windows": True,
                    "report_session_id": None,
                    "capture_product_report": False,
                }
            ],
        )

    def test_view_recent_session_uses_recorded_condition_config_snapshot(self):
        widget = _DummySequenceWidget()
        original_analysis_config = copy.deepcopy(widget.analysis_config)
        original_sequence_config = copy.deepcopy(widget.sequence_config)
        original_using_config_path = widget.using_config_path
        recorded_sequence_config = [
            {
                "seq1": {
                    "acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000}},
                    "analysis_list": {
                        "display_sequence": ["ai_6000"],
                        "ai_6000": {"type": "AI", "config_name": "6000_ai"},
                    },
                }
            }
        ]
        widget._session_record = {
            "recorded_signal_info": {},
            "recorded_path": "history.wav",
            "sample_rate": 48000,
            "condition_key": "q6000",
            "config_snapshot": {
                "sequence_config": recorded_sequence_config,
                "analysis_config": recorded_sequence_config[0]["seq1"]["analysis_list"],
                "using_config_path": "queue_6000.json",
                "active_input_channels": [1],
            },
        }

        widget._show_recent_session_analysis_by_id("recent_1")

        self.assertEqual(widget.run_analysis_configs[0]["display_sequence"], ["ai_6000"])
        self.assertEqual(widget.run_analysis_configs[0]["ai_6000"]["type"], "AI")
        self.assertEqual(widget.run_sequence_configs[0], recorded_sequence_config)
        self.assertEqual(widget._active_input_channels, [0])
        self.assertEqual(widget.analysis_config, original_analysis_config)
        self.assertEqual(widget.sequence_config, original_sequence_config)
        self.assertEqual(widget.using_config_path, original_using_config_path)


class _FailingAnalysisWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.product_test_pdf_report_config = {"enabled": True}
        self._current_recent_session_id = "recent-current"
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self.updated_sessions = []

    def _run_analysis_impl(self, show_windows=True, *, report_session_id=None):
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
        self.recent_session_panel = None
        self.pdf_groups = []

    def _refresh_manual_product_condition_results_from_group(self, _group_id):
        return None

    def _refresh_current_manual_product_final_from_group(self, _group_id):
        return None

    def _try_export_product_test_pdf(self, group_id):
        self.pdf_groups.append(group_id)


class TestProductReportAnalysisFailure(unittest.TestCase):
    def test_analysis_exception_marks_target_session_failed_before_propagating(self):
        widget = _FailingAnalysisWidget()

        with self.assertRaisesRegex(RuntimeError, "analysis crashed"):
            widget.run(show_windows=False)

        self.assertEqual(len(widget.updated_sessions), 1)
        session_id, fields = widget.updated_sessions[0]
        self.assertEqual(session_id, "recent-current")
        self.assertEqual(fields["analysis_report_state"], "failed")
        self.assertEqual(fields["analysis_report_items"][0]["state"], "failed")
        self.assertEqual(fields["analysis_report_items"][0]["error"], "analysis crashed")

    def test_terminal_session_update_uses_pdf_specific_group_evaluation(self):
        widget = _RecentSessionUpdateWidget()

        widget._update_recent_session(
            "recent-single",
            analysis_report_state="completed",
        )

        self.assertEqual(widget.pdf_groups, ["group-single"])


if __name__ == "__main__":
    unittest.main()
