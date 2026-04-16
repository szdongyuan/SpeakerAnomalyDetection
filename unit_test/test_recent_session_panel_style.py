import sys
import types
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QComboBox

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
    @staticmethod
    def _build_left_click_event():
        return QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(1, 1),
            QPointF(1, 1),
            QPointF(1, 1),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_recent_session_table_declares_selected_row_colors(self):
        panel = RecentSessionPanel()
        style = panel.session_table.styleSheet()

        self.assertIn("selection-background-color", style)
        self.assertIn("selection-color", style)
        self.assertIn("QTableWidget::item:selected", style)

    def test_missing_playback_file_shows_information_dialog(self):
        panel = RecentSessionPanel()
        panel.upsert_session(
            {
                "session_id": "recent_1",
                "time_text": "2026-04-16 10:00:00",
                "barcode": "SN001",
                "product_model": "MODEL",
                "mode_text": "正转",
                "result_label": "ok",
                "recorded_signal_info": {},
            }
        )

        panel._refresh_play_button_for_session("recent_1")
        play_btn = panel._get_cell_center_widget(panel.session_table, 0, 5)

        self.assertIsNotNone(play_btn)
        self.assertTrue(play_btn.isEnabled())
        self.assertEqual(play_btn.toolTip(), "当前记录暂无可播放音频")

        with patch("ui.sequence.recent_session_panel.QMessageBox.information") as mock_information:
            panel._on_play_button_clicked("recent_1")

        mock_information.assert_called_once_with(panel, "提示", "当前记录音频文件不可用，无法播放音频。")

    def test_result_column_switches_between_text_and_combo_by_mode(self):
        session_record = {
            "session_id": "recent_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode_text": "正转",
            "result_label": "ok",
            "recorded_signal_info": {"labels": "OK"},
        }
        session_store = {"recent_1": session_record}
        panel = RecentSessionPanel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.upsert_session(session_record)

        self.assertEqual(panel.session_table.item(0, 4).text(), "ok")
        self.assertIsNone(panel.session_table.cellWidget(0, 4))

        panel.set_result_editable(True)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)

        self.assertIsInstance(combo, QComboBox)
        self.assertEqual(combo.currentText(), "OK")

        panel.set_result_editable(False)
        self.assertEqual(panel.session_table.item(0, 4).text(), "ok")
        self.assertIsNone(panel.session_table.cellWidget(0, 4))

    def test_waiting_result_keeps_text_item_in_edit_mode(self):
        session_record = {
            "session_id": "recent_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode_text": "正转",
            "result_label": "等待测试完成",
            "recorded_signal_info": {"labels": "not_labeled"},
        }
        session_store = {"recent_1": session_record}
        panel = RecentSessionPanel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)

        self.assertIsNone(panel.session_table.cellWidget(0, 4))
        self.assertEqual(panel.session_table.item(0, 4).text(), "等待测试完成")
        self.assertEqual(panel.session_table.item(0, 4).toolTip(), "等待测试完成")

    def test_recent_session_column_widths_match_updated_layout(self):
        panel = RecentSessionPanel()

        self.assertEqual(panel.session_table.columnWidth(1), 136)
        self.assertEqual(panel.session_table.columnWidth(4), 138)

    def test_clicking_result_cell_opens_dropdown(self):
        session_record = {
            "session_id": "recent_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode_text": "正转",
            "result_label": "ok",
            "recorded_signal_info": {"labels": "OK"},
        }
        session_store = {"recent_1": session_record}
        panel = RecentSessionPanel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.mousePressEvent(self._build_left_click_event())

        mock_show_popup.assert_called_once()

    def test_clicking_result_text_area_opens_dropdown(self):
        session_record = {
            "session_id": "recent_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode_text": "正转",
            "result_label": "ok",
            "recorded_signal_info": {"labels": "OK"},
        }
        session_store = {"recent_1": session_record}
        panel = RecentSessionPanel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.eventFilter(combo.lineEdit(), self._build_left_click_event())

        mock_show_popup.assert_called_once()

    def test_clicking_result_text_area_does_not_trigger_popup_twice(self):
        session_record = {
            "session_id": "recent_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode_text": "正转",
            "result_label": "ok",
            "recorded_signal_info": {"labels": "OK"},
        }
        session_store = {"recent_1": session_record}
        panel = RecentSessionPanel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.eventFilter(combo.lineEdit(), self._build_left_click_event())
            combo.eventFilter(combo.lineEdit(), QEvent(QEvent.MouseButtonRelease))

        mock_show_popup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
