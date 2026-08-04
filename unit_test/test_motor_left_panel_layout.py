import unittest

from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from base.load_config import LoadUiConfig
from consts import ui_style_const
from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_panel_common import MotorSectionCard
from ui.sequence.motor_summary_panel import MotorSummaryPanel


class TestMotorLeftPanelLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_card_has_blue_title_and_two_sections(self):
        summary_widget = QWidget()
        panel = MotorDetectionLeftPanel(summary_widget)
        scroll_area = panel.layout().itemAt(0).widget()
        content_widget = scroll_area.widget()
        content_layout = content_widget.layout()

        first_widget = content_layout.itemAt(0).widget()
        second_widget = content_layout.itemAt(1).widget()

        self.assertIsInstance(first_widget, MotorAiResultPanel)
        self.assertIsInstance(second_widget, MotorSummaryPanel)

        card = second_widget.findChild(MotorSectionCard)
        self.assertIsInstance(card, MotorSectionCard)

        labels = [lbl.text() for lbl in second_widget.findChildren(QLabel)]
        self.assertIn("信息汇总", labels)
        self.assertNotIn("汇总信息", labels)

        content_scroll_area = card.content_layout.itemAt(0).widget()
        self.assertIn(ui_style_const.COLOR_PANEL_BG, content_scroll_area.styleSheet())
        content_widget = content_scroll_area.widget()
        self.assertEqual(content_widget.objectName(), "motorSectionContent")
        self.assertIn(ui_style_const.COLOR_PANEL_BG, content_widget.styleSheet())
        self.assertIs(content_widget.layout().itemAt(0).widget(), summary_widget)

    def test_ai_result_panel_uses_condition_names(self):
        summary_widget = QWidget()
        condition_configs = LoadUiConfig.load_product_test_program_condition_configs()
        panel = MotorDetectionLeftPanel(summary_widget, condition_configs=condition_configs)
        expected_names = [item["condition_name"] for item in condition_configs]

        self.assertEqual(panel.ai_result_panel.condition_names, expected_names)
        if condition_configs:
            self.assertTrue(panel.set_condition_result(condition_configs[-1]["key"], "NG"))

    def test_ai_result_panel_keeps_rows_when_condition_keys_repeat(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "", "test_queue": "默认配置"},
                {"condition_name": "7000", "trigger_state": "", "test_queue": "3"},
                {"condition_name": "8000", "trigger_state": "", "test_queue": "3"},
            ]
        )

        self.assertEqual(panel.condition_names, ["6000", "7000", "8000"])
        self.assertEqual(len(panel.rows), 3)
        self.assertEqual(len({item["key"] for item in panel.conditions}), 3)

    def test_ai_result_row_status_text_starts_aligned(self):
        short_text = MotorAiResultPanel._row_text("6000", "待检测")
        long_text = MotorAiResultPanel._row_text("10000", "待检测")

        self.assertEqual(short_text.index("待检测"), long_text.index("待检测"))
        self.assertGreater(short_text.index("待检测"), len("  6000        "))

    def test_ai_final_result_stays_near_panel_bottom(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "01"},
                {"condition_name": "7000", "trigger_state": "02"},
            ]
        )
        panel.resize(400, 360)
        panel.show()
        self.app.processEvents()

        content = panel.final_value.parentWidget()
        bottom_gap = content.height() - panel.final_value.geometry().bottom()

        self.assertLessEqual(bottom_gap, 14)

    def test_ai_result_detail_toggles_on_same_condition_click(self):
        condition_configs = LoadUiConfig.load_product_test_program_condition_configs()
        if not condition_configs:
            self.skipTest("no product test conditions configured")
        panel = MotorAiResultPanel(condition_configs=condition_configs)
        key = condition_configs[0]["key"]

        panel.select_condition(key, show_detail=True)
        self.assertFalse(panel.detail_frame.isHidden())

        panel.select_condition(key, show_detail=True)
        self.assertTrue(panel.detail_frame.isHidden())

    def test_ai_result_detail_uses_condition_analysis_config(self):
        panel = MotorAiResultPanel(
            condition_configs=[
                {
                    "condition_name": "6000",
                    "key": "6000",
                    "analysis_list": {
                        "display_sequence": [
                            "声压级 (SPL) 1",
                            "AI 分析 1",
                            "响度分析 1",
                            "频段能量 (FBA) 1",
                        ],
                        "声压级 (SPL) 1": {
                            "type": "SPL",
                            "limit_checked": True,
                            "upper_limit": "100",
                            "lower_limit": "90",
                        },
                        "AI 分析 1": {
                            "type": "AI",
                            "analyse_model_name": "model-a",
                        },
                        "响度分析 1": {
                            "type": "Loudness",
                            "limit_checked": True,
                            "upper_limit": "15",
                            "lower_limit": "8",
                        },
                        "频段能量 (FBA) 1": {
                            "type": "FBA",
                            "limit_checked": True,
                            "upper_limit": "45",
                        },
                    },
                }
            ]
        )

        panel.select_condition("6000", show_detail=True)
        detail_text = "\n".join(label.text() for label in panel.detail_frame.findChildren(QLabel))

        self.assertEqual(list(panel.detail_labels.keys()), ["SPL", "响度", "FBA"])
        self.assertIn("SPL", detail_text)
        self.assertIn("阈值 90 ~ 100 dB", detail_text)
        self.assertIn("响度", detail_text)
        self.assertIn("阈值 8 ~ 15 sone", detail_text)
        self.assertIn("FBA", detail_text)
        self.assertIn("上限 45 dB", detail_text)
        self.assertNotIn("AI", detail_text)
        self.assertNotIn("model-a", detail_text)
        self.assertNotIn("71.6", detail_text)


if __name__ == "__main__":
    unittest.main()
