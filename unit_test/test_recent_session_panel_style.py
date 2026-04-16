import sys
import types
import unittest

from PyQt5.QtWidgets import QApplication

if "base.playback_controller" not in sys.modules:
    playback_controller = types.ModuleType("base.playback_controller")

    class _PlaybackController:
        def is_audio_playing(self):
            return False

        def stop_audio_playback(self):
            return None

        def get_current_playing_file(self):
            return None

    playback_controller.PlaybackController = _PlaybackController
    sys.modules["base.playback_controller"] = playback_controller

from ui.sequence.recent_session_panel import RecentSessionPanel


class TestRecentSessionPanelStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_recent_session_table_declares_selected_row_colors(self):
        panel = RecentSessionPanel()
        style = panel.session_table.styleSheet()

        self.assertIn("selection-background-color", style)
        self.assertIn("selection-color", style)
        self.assertIn("QTableWidget::item:selected", style)


if __name__ == "__main__":
    unittest.main()
