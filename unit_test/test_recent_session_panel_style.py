import sys
import types
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QComboBox, QGraphicsOpacityEffect, QLabel, QToolButton

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
    def _condition_configs():
        return [
            {"key": "01", "trigger_state": "01", "condition_name": "6000 rpm"},
            {"key": "02", "trigger_state": "02", "condition_name": "7000 rpm"},
        ]

    def _panel(self, **kwargs):
        kwargs.setdefault("condition_configs", self._condition_configs())
        return RecentSessionPanel(**kwargs)

    @staticmethod
    def _session(session_id, condition_key, result_label="ok"):
        return {
            "session_id": session_id,
            "group_id": "group_1",
            "time_text": "2026-04-16 10:00:00",
            "barcode": "SN001",
            "product_model": "MODEL",
            "mode": condition_key,
            "condition_key": condition_key,
            "mode_text": f"{condition_key} rpm",
            "result_label": result_label,
            "recorded_signal_info": {"labels": result_label},
        }

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

    @staticmethod
    def _summary_value(panel, row=0):
        item = panel.session_table.item(row, 3 + len(panel.conditions))
        return item.data(Qt.UserRole) if item is not None else None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_recent_session_table_declares_selected_row_colors(self):
        panel = self._panel()
        style = panel.session_table.styleSheet()

        self.assertIn("selection-background-color", style)
        self.assertIn("selection-color", style)
        self.assertIn("QTableWidget::item:selected", style)

    def test_recent_session_table_uses_dynamic_condition_columns(self):
        panel = self._panel()
        panel.upsert_session(self._session("recent_1", "01", "ok"))

        self.assertEqual(panel.session_table.columnCount(), 6)
        self.assertEqual(panel.session_table.horizontalHeaderItem(3).text(), "6000 rpm")
        self.assertEqual(panel.session_table.horizontalHeaderItem(4).text(), "7000 rpm")
        self.assertEqual(panel.session_table.horizontalHeaderItem(5).text(), "汇总结果")
        self.assertIsNone(panel._get_cell_center_widget(panel.session_table, 0, 6))

    def test_condition_headers_append_rpm_when_missing(self):
        panel = self._panel(
            condition_configs=[
                {"key": "01", "trigger_state": "01", "condition_name": "6000"},
                {"key": "02", "trigger_state": "02", "condition_name": "7000"},
            ]
        )

        self.assertEqual(panel.session_table.horizontalHeaderItem(3).text(), "6000 rpm")
        self.assertEqual(panel.session_table.horizontalHeaderItem(4).text(), "7000 rpm")

    def test_result_cells_are_dropdowns(self):
        session_record = self._session("recent_1", "01", "OK")
        session_store = {"recent_1": session_record}
        panel = self._panel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.upsert_session(session_record)

        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)

        self.assertIsInstance(combo, QComboBox)
        self.assertEqual(combo.currentText(), "OK")

    def test_condition_cells_include_analysis_button(self):
        clicked_sessions = []
        session_record = self._session("recent_1", "01", "OK")
        panel = self._panel(on_view_session=lambda session_id: clicked_sessions.append(session_id))
        panel.upsert_session(session_record)

        cell_widget = panel.session_table.cellWidget(0, 3)
        view_btn = cell_widget.layout().itemAt(1).widget()

        self.assertIsInstance(view_btn, QToolButton)
        self.assertTrue(view_btn.isEnabled())
        view_btn.click()
        self.assertEqual(clicked_sessions, ["recent_1"])

    def test_fixture_order_keeps_one_group_and_each_condition_analysis_selectable(self):
        clicked_sessions = []
        panel = self._panel(
            on_view_session=lambda session_id: clicked_sessions.append(session_id)
        )

        panel.upsert_session(self._session("recent_7000", "02", "OK"))
        panel.upsert_session(self._session("recent_6000", "01", "NG"))

        self.assertEqual(panel.session_table.rowCount(), 1)
        self.assertEqual(
            panel.group_records["group_1"]["session_ids"],
            {"02": "recent_7000", "01": "recent_6000"},
        )
        for column in (3, 4):
            cell_widget = panel.session_table.cellWidget(0, column)
            view_btn = cell_widget.layout().itemAt(1).widget()
            self.assertTrue(view_btn.isEnabled())
            view_btn.click()

        self.assertEqual(clicked_sessions, ["recent_6000", "recent_7000"])

    def test_condition_analysis_button_disabled_without_record(self):
        panel = self._panel()
        panel.upsert_session(self._session("recent_1", "01", "OK"))

        cell_widget = panel.session_table.cellWidget(0, 4)
        view_btn = cell_widget.layout().itemAt(1).widget()

        self.assertIsInstance(view_btn, QToolButton)
        self.assertFalse(view_btn.isEnabled())
        self.assertIsInstance(view_btn.graphicsEffect(), QGraphicsOpacityEffect)
        self.assertLessEqual(view_btn.graphicsEffect().opacity(), 0.35)

    def test_panel_group_only_keeps_session_ids_not_analysis_images(self):
        panel = self._panel()
        session_record = self._session("recent_1", "01", "OK")
        session_record["analysis_report_items"] = [
            {
                "name": "声压级",
                "images": [{"png_data": b"large-png-data"}],
            }
        ]

        panel.upsert_session(session_record)

        self.assertNotIn(
            "analysis_report_items",
            panel.session_record_by_id["recent_1"],
        )
        group = panel.group_records["group_1"]
        self.assertEqual(group["session_ids"], {"01": "recent_1"})
        self.assertNotIn("records", group)

    def test_summary_result_is_ng_when_any_condition_is_ng(self):
        panel = self._panel()

        panel.upsert_session(self._session("recent_1", "01", "OK"))
        panel.upsert_session(self._session("recent_2", "02", "NG"))

        self.assertEqual(panel.session_table.rowCount(), 1)
        self.assertEqual(panel._get_cell_center_widget(panel.session_table, 0, 3).currentText(), "OK")
        self.assertEqual(panel._get_cell_center_widget(panel.session_table, 0, 4).currentText(), "NG")
        self.assertIn("#166534", panel._get_cell_center_widget(panel.session_table, 0, 3).styleSheet())
        self.assertIn("#991B1B", panel._get_cell_center_widget(panel.session_table, 0, 4).styleSheet())
        self.assertEqual(self._summary_value(panel), "NG")
        summary_widget = panel.session_table.cellWidget(0, 5).layout().itemAt(0).widget()
        self.assertIsInstance(summary_widget, QLabel)
        self.assertEqual(summary_widget.text(), "NG")
        self.assertEqual(panel.session_table.item(0, 5).text(), "")
        self.assertIn("#991B1B", summary_widget.styleSheet())
        self.assertIn("border: none", summary_widget.styleSheet())

    def test_summary_result_is_ok_only_when_all_conditions_are_ok(self):
        panel = self._panel()

        panel.upsert_session(self._session("recent_1", "01", "OK"))
        self.assertEqual(self._summary_value(panel), "not_labeled")

        panel.upsert_session(self._session("recent_2", "02", "OK"))
        self.assertEqual(self._summary_value(panel), "OK")
        summary_widget = panel.session_table.cellWidget(0, 5).layout().itemAt(0).widget()
        self.assertEqual(summary_widget.text(), "OK")
        self.assertIn("#166534", summary_widget.styleSheet())
        self.assertIn("background: transparent", summary_widget.styleSheet())

    def test_not_labeled_results_display_as_chinese_text(self):
        panel = self._panel()
        panel.upsert_session(self._session("recent_1", "01", "not_labeled"))

        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)
        summary_widget = panel.session_table.cellWidget(0, 5).layout().itemAt(0).widget()
        option_texts = [combo.itemText(index) for index in range(combo.count())]

        self.assertEqual(combo.currentText(), "未标记")
        self.assertIn("未标记", option_texts)
        self.assertNotIn("not_labeled", option_texts)
        self.assertEqual(combo.itemData(option_texts.index("OK"), Qt.ForegroundRole).name().upper(), "#166534")
        self.assertEqual(combo.itemData(option_texts.index("NG"), Qt.ForegroundRole).name().upper(), "#991B1B")
        self.assertEqual(combo.itemData(option_texts.index("未标记"), Qt.ForegroundRole).name().upper(), "#26364A")
        self.assertEqual(self._summary_value(panel), "not_labeled")
        self.assertEqual(summary_widget.text(), "未标记")
        self.assertIn("#475569", summary_widget.styleSheet())
        self.assertEqual(panel.session_table.item(0, 5).toolTip(), "未标记")

    def test_selecting_chinese_not_labeled_passes_internal_label_to_callback(self):
        changed_labels = []
        panel = self._panel(
            on_change_session_result=lambda session_id, label: changed_labels.append((session_id, label)) or True
        )
        panel.set_result_editable(True)
        panel.upsert_session(self._session("recent_1", "01", "OK"))

        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)
        combo.setCurrentText("未标记")

        self.assertEqual(changed_labels[-1], ("recent_1", "not_labeled"))

    def test_sessions_without_group_id_create_separate_rows(self):
        panel = self._panel()
        first = self._session("recent_1", "01", "OK")
        second = self._session("recent_2", "02", "NG")
        first.pop("group_id", None)
        second.pop("group_id", None)

        panel.upsert_session(first)
        panel.upsert_session(second)

        self.assertEqual(panel.session_table.rowCount(), 2)

    def test_reset_sessions_ignores_removed_play_column(self):
        conditions = [
            {"key": "01", "trigger_state": "01", "condition_name": "6000"},
            {"key": "02", "trigger_state": "02", "condition_name": "7000"},
            {"key": "03", "trigger_state": "03", "condition_name": "8000"},
            {"key": "04", "trigger_state": "04", "condition_name": "9000"},
        ]
        panel = self._panel(condition_configs=conditions)
        for index, condition in enumerate(conditions, start=1):
            panel.upsert_session(self._session(f"recent_{index}", condition["key"], "OK"))

        panel.reset_sessions()

        self.assertEqual(panel.session_table.rowCount(), 0)

    def test_selecting_condition_dropdown_updates_summary(self):
        panel = self._panel()
        panel.upsert_session(self._session("recent_1", "01", "OK"))
        panel.upsert_session(self._session("recent_2", "02", "OK"))

        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)
        combo.setCurrentText("NG")

        self.assertEqual(self._summary_value(panel), "NG")

    def test_unrecorded_condition_result_dropdown_is_blocked(self):
        changed_labels = []
        panel = self._panel(
            on_change_session_result=lambda session_id, label: changed_labels.append((session_id, label)) or True
        )
        panel.set_result_editable(True)
        panel.upsert_session(self._session("recent_1", "01", "OK"))

        combo = panel._get_cell_center_widget(panel.session_table, 0, 4)
        self.assertEqual(combo.currentText(), "未标记")

        with patch("ui.sequence.recent_session_panel.QMessageBox.warning") as warning:
            combo.setCurrentText("NG")

        warning.assert_called_once()
        self.assertIn("录音尚未完成", warning.call_args[0][2])
        self.assertEqual(combo.currentText(), "未标记")
        self.assertEqual(self._summary_value(panel), "not_labeled")
        self.assertEqual(changed_labels, [])

    def test_recent_session_column_widths_match_updated_layout(self):
        from PyQt5.QtWidgets import QHeaderView

        panel = self._panel()
        header = panel.session_table.horizontalHeader()

        panel.resize(1400, 420)
        panel.show()
        self.addCleanup(panel.close)
        self.app.processEvents()

        self.assertEqual(panel.session_table.columnCount(), 6)
        for col in range(panel.session_table.columnCount()):
            self.assertEqual(header.sectionResizeMode(col), QHeaderView.Stretch)
        total_width = sum(panel.session_table.columnWidth(col) for col in range(panel.session_table.columnCount()))
        self.assertGreaterEqual(total_width, panel.session_table.viewport().width() - 2)
        self.assertLessEqual(total_width, panel.session_table.viewport().width() + panel.session_table.columnCount())
        self.assertGreater(panel.session_table.columnWidth(3), 150)

    def test_clicking_result_cell_opens_dropdown(self):
        session_record = self._session("recent_1", "01", "OK")
        session_store = {"recent_1": session_record}
        panel = self._panel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.mousePressEvent(self._build_left_click_event())

        mock_show_popup.assert_called_once()

    def test_clicking_result_text_area_opens_dropdown(self):
        session_record = self._session("recent_1", "01", "OK")
        session_store = {"recent_1": session_record}
        panel = self._panel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.eventFilter(combo.lineEdit(), self._build_left_click_event())

        mock_show_popup.assert_called_once()

    def test_clicking_result_text_area_does_not_trigger_popup_twice(self):
        session_record = self._session("recent_1", "01", "OK")
        session_store = {"recent_1": session_record}
        panel = self._panel(on_play_session=lambda session_id: session_store.get(session_id))
        panel.set_result_editable(True)
        panel.upsert_session(session_record)
        combo = panel._get_cell_center_widget(panel.session_table, 0, 3)

        with patch.object(combo, "showPopup") as mock_show_popup:
            combo.eventFilter(combo.lineEdit(), self._build_left_click_event())
            combo.eventFilter(combo.lineEdit(), QEvent(QEvent.MouseButtonRelease))

        mock_show_popup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
