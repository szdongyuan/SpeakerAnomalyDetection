import unittest

from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_mode_switch_panel import MotorModeSwitchPanel
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

        card = second_widget.layout().itemAt(0).widget()
        self.assertIsInstance(card, MotorSectionCard)

        labels = [lbl.text() for lbl in second_widget.findChildren(QLabel)]
        self.assertIn("操作面板", labels)
        self.assertIn("模式切换", labels)
        self.assertIn("汇总信息", labels)

        embedded_mode_switch = card.content_layout.itemAt(1).widget()
        self.assertIsInstance(embedded_mode_switch, MotorModeSwitchPanel)


if __name__ == "__main__":
    unittest.main()
