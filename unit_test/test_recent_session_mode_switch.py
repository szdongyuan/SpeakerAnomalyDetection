import unittest
import logging
import sys
import types
from unittest.mock import patch

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

from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _DummyRecentSessionPanel:
    def __init__(self):
        self.editable_states = []
        self.reset_count = 0
        self.conditions = None

    def set_result_editable(self, editable: bool):
        self.editable_states.append(bool(editable))

    def reset_sessions(self):
        self.reset_count += 1

    def set_conditions(self, condition_configs):
        self.conditions = list(condition_configs or [])
        self.reset_sessions()


class _DummySequenceWidget(SequenceWidgetStreamingOpsMixin):
    def __init__(self):
        self.recent_session_panel = _DummyRecentSessionPanel()
        self.recent_test_sessions = ["recent_1", "recent_2"]
        self.recent_test_session_by_id = {"recent_1": {"session_id": "recent_1"}, "recent_2": {"session_id": "recent_2"}}
        self._current_recent_session_id = "recent_2"
        self._pending_recent_session_append = True
        self._last_recent_session_mode = "mark"
        self.product_test_condition_configs = [{"key": "01", "condition_name": "6000"}]

    def _clear_recent_session_history(self, reset_panel=True):
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self._current_recent_session_id = None
        self._pending_recent_session_append = False
        if reset_panel:
            self.recent_session_panel.reset_sessions()


class TestRecentSessionModeSwitch(unittest.TestCase):
    def test_mode_switch_clears_recent_history(self):
        widget = _DummySequenceWidget()

        widget._on_recent_session_mode_changed({"mode": "test"})

        self.assertEqual(widget.recent_session_panel.editable_states, [False])
        self.assertEqual(widget.recent_session_panel.reset_count, 1)
        self.assertEqual(widget.recent_test_sessions, [])
        self.assertEqual(widget.recent_test_session_by_id, {})
        self.assertIsNone(widget._current_recent_session_id)
        self.assertFalse(widget._pending_recent_session_append)
        self.assertEqual(widget._last_recent_session_mode, "test")

    def test_same_mode_update_does_not_clear_recent_history(self):
        widget = _DummySequenceWidget()

        widget._on_recent_session_mode_changed({"mode": "mark"})

        self.assertEqual(widget.recent_session_panel.editable_states, [True])
        self.assertEqual(widget.recent_session_panel.reset_count, 0)
        self.assertEqual(widget.recent_test_sessions, ["recent_1", "recent_2"])
        self.assertEqual(widget._last_recent_session_mode, "mark")

    def test_config_sync_keeps_recent_history_when_conditions_do_not_change(self):
        widget = _DummySequenceWidget()
        condition_configs = [{"key": "01", "condition_name": "6000"}]

        with patch(
            "ui.sequence.sequence_widget_streaming_ops.LoadUiConfig.load_product_test_program_condition_configs",
            return_value=condition_configs,
        ):
            widget._sync_product_test_conditions()

        self.assertEqual(widget.recent_test_sessions, ["recent_1", "recent_2"])
        self.assertEqual(widget.recent_session_panel.reset_count, 0)
        self.assertIsNone(widget.recent_session_panel.conditions)

    def test_config_sync_clears_recent_history_when_conditions_change(self):
        widget = _DummySequenceWidget()
        condition_configs = [{"key": "02", "condition_name": "7000"}]

        with patch(
            "ui.sequence.sequence_widget_streaming_ops.LoadUiConfig.load_product_test_program_condition_configs",
            return_value=condition_configs,
        ):
            widget._sync_product_test_conditions()

        self.assertEqual(widget.recent_test_sessions, [])
        self.assertEqual(widget.recent_test_session_by_id, {})
        self.assertIsNone(widget._current_recent_session_id)
        self.assertFalse(widget._pending_recent_session_append)
        self.assertEqual(widget.recent_session_panel.reset_count, 1)
        self.assertEqual(widget.recent_session_panel.conditions, condition_configs)

    def test_config_switch_forces_recent_history_clear(self):
        widget = _DummySequenceWidget()
        condition_configs = [{"key": "01", "condition_name": "6000"}]

        with patch(
            "ui.sequence.sequence_widget_streaming_ops.LoadUiConfig.load_product_test_program_condition_configs",
            return_value=condition_configs,
        ):
            widget._sync_product_test_conditions(clear_recent_history=True)

        self.assertEqual(widget.recent_test_sessions, [])
        self.assertEqual(widget.recent_session_panel.reset_count, 1)
        self.assertEqual(widget.recent_session_panel.conditions, condition_configs)


if __name__ == "__main__":
    unittest.main()
