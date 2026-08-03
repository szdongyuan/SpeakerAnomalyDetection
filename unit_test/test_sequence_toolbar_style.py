import unittest

from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from consts import ui_style_const
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin


class _DummySerialStatusWidget(QWidget, SequenceWidgetSerialTriggerOpsMixin):
    def __init__(self):
        super().__init__()
        self.serial_trigger_status_label = QLabel()


class TestSequenceToolbarStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_toolbar_buttons_keep_accessibility_text(self):
        toolbar = SequenceToolsBar()
        expectations = [
            (toolbar.player_btn, "开始录制"),
            (toolbar.replayer_btn, "重新录制"),
            (toolbar.data_btn, "分析"),
            (toolbar.tcp_btn, "tcp配置"),
            (toolbar.serial_trigger_btn, "串口离散输入触发配置"),
        ]

        for button, expected in expectations:
            with self.subTest(expected=expected):
                self.assertEqual(button.toolTip(), expected)
                self.assertEqual(button.accessibleName(), expected)
                self.assertEqual(button.accessibleDescription(), expected)
                self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, button.styleSheet())

    def test_toolbar_uses_light_blue_container_style(self):
        toolbar = SequenceToolsBar()

        self.assertEqual(toolbar.objectName(), "sequenceToolsBar")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BG, toolbar.styleSheet())

    def test_toolbar_has_condition_mode_combobox(self):
        toolbar = SequenceToolsBar()

        self.assertEqual(toolbar.condition_mode_combobox.itemText(0), "测试")
        self.assertEqual(toolbar.condition_mode_combobox.itemText(1), "标记")
        self.assertIn(ui_style_const.COLOR_BORDER_STRONG, toolbar.condition_mode_combobox.styleSheet())

    def test_toolbar_comboboxes_show_dropdown_arrow(self):
        toolbar = SequenceToolsBar()

        self.assertIn("QComboBox::down-arrow", toolbar.using_file_combobox.styleSheet())
        self.assertIn("QComboBox::down-arrow", toolbar.condition_mode_combobox.styleSheet())
        self.assertIn(ui_style_const.COMBO_DOWN_ARROW_ICON, toolbar.using_file_combobox.styleSheet())

    def test_using_config_combobox_refreshes_before_popup(self):
        toolbar = SequenceToolsBar()
        calls = []
        toolbar.using_file_combobox.before_show_popup = lambda: calls.append("refresh")

        toolbar.using_file_combobox.showPopup()
        self.app.processEvents()
        toolbar.using_file_combobox.hidePopup()

        self.assertEqual(calls, ["refresh"])

    def test_serial_status_badge_updates_style_with_connection_state(self):
        widget = _DummySerialStatusWidget()

        widget.on_serial_trigger_status_changed({"connected": False, "has_response": False, "message": "off"})
        self.assertEqual(widget.serial_trigger_status_label.text(), "未连接")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, widget.serial_trigger_status_label.styleSheet())
        self.assertIn(ui_style_const.COLOR_TEXT_MUTED, widget.serial_trigger_status_label.styleSheet())

        widget.on_serial_trigger_status_changed({"connected": True, "has_response": False, "message": "open"})
        self.assertEqual(widget.serial_trigger_status_label.text(), "已打开")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, widget.serial_trigger_status_label.styleSheet())
        self.assertIn(ui_style_const.COLOR_PRIMARY_HOVER, widget.serial_trigger_status_label.styleSheet())

        widget.on_serial_trigger_status_changed({"connected": True, "has_response": True, "message": "ok"})
        self.assertEqual(widget.serial_trigger_status_label.text(), "已连接")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, widget.serial_trigger_status_label.styleSheet())
        self.assertIn(ui_style_const.COLOR_OK, widget.serial_trigger_status_label.styleSheet())


if __name__ == "__main__":
    unittest.main()
