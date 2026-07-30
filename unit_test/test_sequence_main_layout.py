import logging
import sys
import types
import unittest

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplitter, QVBoxLayout, QWidget

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_panel_common import MotorSectionCard
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class _DummyCountBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = "test"
        self.mode_change_callbacks = []

    def register_mode_change_callback(self, callback):
        self.mode_change_callbacks.append(callback)


class _DummyChannelWorkspace:
    def __init__(self, keys):
        self._keys = list(keys)
        self.results = []
        self.audio_paths = []

    def condition_keys(self):
        return list(self._keys)

    def set_condition_result(self, key, label):
        self.results.append((key, label))

    def set_condition_audio_path(self, key, path):
        self.audio_paths.append((key, path))


class _SpyLeftPanel:
    def __init__(self):
        self.results = []

    def set_condition_result(self, key, label, tone=None):
        self.results.append((key, label, tone))


class _DummySequenceWidget(QWidget, SequenceWidgetStreamingOpsMixin):
    def __init__(self):
        super().__init__()
        self.count_board = _DummyCountBoard()
        self.left_panel = MotorDetectionLeftPanel(self.count_board)
        self.channel_workspace = None
        self.recent_session_panel = None
        self._last_recent_session_mode = ""

    def _resolve_recent_session(self, session_id: str):
        return None

    def _show_recent_session_analysis_by_id(self, session_id: str):
        return None

    def _change_recent_session_result_by_id(self, session_id: str, label: str):
        return None

    def _configure_direction_waveform_workspace(self):
        return None

    def _on_recent_session_mode_changed(self, mode: str):
        self._last_recent_session_mode = mode


class _RealisticSequenceWidget(_DummySequenceWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.count_board = SequenceCountBoard({})
        self.left_panel = MotorDetectionLeftPanel(self.count_board)
        self.channel_workspace = None
        self.recent_session_panel = None
        self._last_recent_session_mode = ""


class TestSequenceMainLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_waveform_layout_uses_top_bottom_split_with_two_horizontal_rows(self):
        widget = _DummySequenceWidget()

        layout = widget.create_waveform_layout()

        self.assertIsInstance(layout, QVBoxLayout)
        self.assertEqual(layout.count(), 1)

        main_splitter = layout.itemAt(0).widget()
        self.assertIsInstance(main_splitter, QSplitter)
        self.assertEqual(main_splitter.orientation(), Qt.Vertical)
        self.assertEqual(main_splitter.count(), 2)

        top_row_splitter = main_splitter.widget(0)
        bottom_row_splitter = main_splitter.widget(1)
        self.assertIsInstance(top_row_splitter, QSplitter)
        self.assertIsInstance(bottom_row_splitter, QSplitter)
        self.assertEqual(top_row_splitter.orientation(), Qt.Horizontal)
        self.assertEqual(bottom_row_splitter.orientation(), Qt.Horizontal)

        self.assertIs(top_row_splitter.widget(0), widget.left_panel.ai_result_panel)
        self.assertIs(top_row_splitter.widget(1), widget.channel_workspace)
        self.assertIs(bottom_row_splitter.widget(0), widget.left_panel.summary_panel)
        self.assertIs(bottom_row_splitter.widget(1), widget.recent_session_panel)

    def test_common_window_size_keeps_waveform_and_history_readable(self):
        widget = _DummySequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        self.assertGreaterEqual(widget.channel_workspace.width(), 700)
        self.assertGreaterEqual(widget.channel_workspace.height(), 360)
        self.assertGreaterEqual(widget.recent_session_panel.width(), 700)
        self.assertGreaterEqual(widget.recent_session_panel.height(), 300)
        self.assertGreater(widget.channel_workspace.width(), widget.left_panel.ai_result_panel.width())
        self.assertGreater(widget.recent_session_panel.width(), widget.left_panel.summary_panel.width())

    def test_real_operation_panel_keeps_minimum_height_within_window_budget(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 700)
        widget.show()
        self.app.processEvents()

        self.assertLessEqual(widget.minimumSizeHint().height(), 700)

    def test_left_cards_stretch_to_match_their_row_heights(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        ai_card = widget.left_panel.ai_result_panel.findChild(MotorSectionCard)
        summary_card = widget.left_panel.summary_panel.findChild(MotorSectionCard)

        self.assertIsNotNone(ai_card)
        self.assertIsNotNone(summary_card)
        self.assertEqual(ai_card.height(), widget.left_panel.ai_result_panel.height())
        self.assertEqual(summary_card.height(), widget.left_panel.summary_panel.height())

        recent_card = widget.recent_session_panel.layout().itemAt(0).widget()
        recent_title = recent_card.layout().itemAt(0).widget()
        summary_title = summary_card.layout().itemAt(0).widget()
        self.assertEqual(recent_title.sizeHint().height(), summary_title.sizeHint().height())

    def test_default_main_splitter_ratio_is_45_to_55(self):
        widget = _RealisticSequenceWidget()
        layout = widget.create_waveform_layout()
        widget.setLayout(layout)
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        main_splitter = layout.itemAt(0).widget()
        top_height, bottom_height = main_splitter.sizes()
        total_height = top_height + bottom_height

        self.assertGreater(total_height, 0)
        self.assertAlmostEqual(top_height / total_height, 0.45, delta=0.03)
        self.assertAlmostEqual(bottom_height / total_height, 0.55, delta=0.03)

    def test_count_board_keeps_test_summary_visible_in_mark_mode(self):
        board = SequenceCountBoard({})

        board.on_mark_btn_clicked()

        self.assertEqual(board.mode, "mark")
        self.assertEqual(board.stacked_widget.currentIndex(), 0)

        board.on_test_btn_clicked()

        self.assertEqual(board.mode, "test")
        self.assertEqual(board.stacked_widget.currentIndex(), 0)

    def test_waveform_condition_actions_follow_mode(self):
        played = []
        marked = []
        panel = DirectionWaveformPanel(
            condition_configs=[{"condition_name": "6000 rpm", "trigger_state": "01"}],
            on_play_condition=played.append,
            on_mark_condition=lambda key, label: marked.append((key, label)),
        )
        card = panel._cards["01"]
        margins = panel.grid.contentsMargins()

        self.assertFalse(card.play_btn.isHidden())
        self.assertTrue(card.mark_panel.isHidden())
        self.assertEqual((margins.left(), margins.top(), margins.right(), margins.bottom()), (8, 0, 8, 8))

        panel.set_mode("mark")
        self.assertFalse(card.mark_panel.isHidden())

        card.play_btn.click()
        card.ok_btn.click()

        self.assertEqual(played, ["01"])
        self.assertEqual(marked, [("01", "OK")])

    def test_waveform_mark_does_not_update_left_condition_judgement(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01"])
        widget.left_panel = _SpyLeftPanel()
        widget._condition_record_cache = {}
        widget.recorded_path = None
        widget.recorded_signal_info = {}

        widget.on_waveform_condition_mark_clicked("01", "OK")

        self.assertEqual(widget.channel_workspace.results, [("01", "OK")])
        self.assertEqual(widget.left_panel.results, [])


if __name__ == "__main__":
    unittest.main()
