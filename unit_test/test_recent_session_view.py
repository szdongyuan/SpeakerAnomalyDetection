import logging
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

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


class _DummySequenceWidget(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self._session_record = {"recorded_signal_info": {}, "sample_rate": 48000}
        self.recorded_path = "current.wav"
        self.recorded_signal_info = {"file_path": "current.wav"}
        self.data_struct = SimpleNamespace(
            store_wave_data=np.array([1.0, 2.0], dtype=np.float32),
            store_wave_data_multi=np.array([[1.0], [2.0]], dtype=np.float32),
            sample_rate=48000,
            audio_lenth=2,
            analysis_result_dict={"legacy": (True, 0.0)},
        )
        self.count_board = SimpleNamespace(mode="test")
        self._excel_export_cache = {"cached": True}
        self._excel_exported_record_id = "current.wav"
        self.analysis_config = {"auto_analysis": True}
        self.data_btn = _DummyButton()
        self.run_calls = []
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

    def run(self, show_windows=True):
        self.run_calls.append(show_windows)

    def plot_waveform_to_workspace(self, data, sample_rate):
        self.plot_calls.append((np.asarray(data).copy(), sample_rate))

    def _clear_plot_area(self):
        self.clear_plot_calls += 1


class TestRecentSessionView(unittest.TestCase):
    def test_view_recent_session_does_not_trigger_second_silent_run(self):
        widget = _DummySequenceWidget()

        widget._show_recent_session_analysis_by_id("recent_1")

        self.assertEqual(widget.run_calls, [True])


if __name__ == "__main__":
    unittest.main()
