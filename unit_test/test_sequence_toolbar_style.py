import unittest

from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from consts import ui_style_const
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin


class _DummySerialStatusWidget(QWidget, SequenceWidgetSerialTriggerOpsMixin):
    def __init__(self):
        super().__init__()
        self.serial_trigger_btn = QPushButton()


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

    def test_replay_button_is_retained_but_hidden(self):
        toolbar = SequenceToolsBar()

        self.assertTrue(toolbar.replayer_btn.isHidden())
        self.assertFalse(toolbar.replayer_btn.isEnabled())
        self.assertEqual(toolbar.replayer_btn.toolTip(), "重新录制")

    def test_product_fields_follow_action_buttons_before_flexible_space(self):
        toolbar = SequenceToolsBar()
        main_layout = toolbar.layout().itemAt(1).layout()
        serial_button_index = main_layout.indexOf(toolbar.serial_trigger_btn)

        self.assertGreaterEqual(serial_button_index, 0)
        self.assertIsNotNone(main_layout.itemAt(serial_button_index + 2).layout())

    def test_analysis_button_uses_static_progress_and_terminal_badges(self):
        toolbar = SequenceToolsBar()
        button = toolbar.data_btn

        button.set_analyzing(7, 20, "A口 / 0.1")
        self.assertEqual(button.analysis_state, button.STATE_ANALYZING)
        self.assertEqual(button.status_badge.text(), "7/20")
        self.assertFalse(button.status_badge.isHidden())
        self.assertIn("A口 / 0.1", button.toolTip())
        self.assertTrue(button.rect().contains(button.status_badge.geometry()))

        button.set_completed("A口 / 0.1")
        self.assertEqual(button.analysis_state, button.STATE_COMPLETED)
        self.assertEqual(button.status_badge.text(), "✓")
        self.assertIn("点击查看", button.toolTip())

        button.set_failed("A口 / 0.1")
        self.assertEqual(button.analysis_state, button.STATE_FAILED)
        self.assertEqual(button.status_badge.text(), "!")
        self.assertIn("查看原因", button.toolTip())

        button.set_idle()
        self.assertEqual(button.analysis_state, button.STATE_IDLE)
        self.assertTrue(button.status_badge.isHidden())
        self.assertEqual(button.toolTip(), "分析")

    def test_toolbar_uses_light_blue_container_style(self):
        toolbar = SequenceToolsBar()

        self.assertEqual(toolbar.objectName(), "sequenceToolsBar")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BG, toolbar.styleSheet())

    def test_main_ui_uses_simsun_except_for_small_auxiliary_text(self):
        self.assertEqual(ui_style_const.UI_FONT_FAMILY_NAME, "SimSun")
        self.assertEqual(ui_style_const.UI_FONT_FAMILY, "'SimSun'")
        self.assertEqual(
            ui_style_const.MAIN_UI_SMALL_FONT_FAMILY_NAME,
            "Microsoft YaHei UI",
        )
        self.assertEqual(
            ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
            "'Microsoft YaHei UI'",
        )
        default_font_styles = (
            ui_style_const.main_window_base_style,
            ui_style_const.main_window_title_label_style,
            ui_style_const.toolbar_button_style,
            ui_style_const.toolbar_input_style,
            ui_style_const.toolbar_spinbox_style,
            ui_style_const.toolbar_combobox_style,
            ui_style_const.serial_trigger_button_base_style,
        )
        for style in default_font_styles:
            with self.subTest(style=style[:40]):
                self.assertIn(ui_style_const.UI_FONT_FAMILY, style)
                self.assertNotIn(
                    ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
                    style,
                )

    def test_menu_and_toolbar_labels_share_bold_legacy_font(self):
        menu_style = ui_style_const.main_window_menubar_style
        self.assertIn(ui_style_const.UI_FONT_FAMILY, menu_style)
        self.assertNotIn(ui_style_const.MAIN_UI_SMALL_FONT_FAMILY, menu_style)
        self.assertEqual(menu_style.count("font-size: 18px"), 3)
        self.assertIn("font-weight: 600", menu_style)
        self.assertIn("padding-top: 3px", menu_style)

        for style in (
            ui_style_const.toolbar_field_label_style,
            ui_style_const.toolbar_checkbox_style,
        ):
            with self.subTest(style=style[:40]):
                self.assertIn(ui_style_const.UI_FONT_FAMILY, style)
                self.assertNotIn(
                    ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
                    style,
                )
                self.assertIn("font-size: 18px", style)
                self.assertIn("font-weight: 600", style)

    def test_statusbar_uses_muted_regular_text(self):
        status_styles = (
            ui_style_const.main_window_statusbar_style,
            ui_style_const.main_window_status_label_style,
        )

        for style in status_styles:
            with self.subTest(style=style[:40]):
                self.assertIn(ui_style_const.COLOR_TEXT_MUTED, style)
                self.assertIn(
                    ui_style_const.MAIN_UI_SMALL_FONT_FAMILY,
                    style,
                )
                self.assertIn("font-size: 15px", style)
                self.assertIn("font-weight: 400", style)

    def test_splash_uses_the_yahei_ui_font(self):
        from ui.splash_screen_window import Splash

        splash = Splash()
        style = splash.product_name_label.styleSheet()
        status_style = splash.lab.styleSheet()

        self.assertIn(ui_style_const.MAIN_UI_SMALL_FONT_FAMILY, style)
        self.assertIn("font-size: 22px", style)
        self.assertIn(ui_style_const.MAIN_UI_SMALL_FONT_FAMILY, status_style)
        self.assertIn("font-size: 12px", status_style)
        self.assertEqual(splash.lab.width(), 320)
        self.assertEqual(splash.lab.y(), 344)
        self.assertEqual(splash.prg.y(), 366)
        self.assertNotIn("STXingkai", style)
        self.assertNotIn("KaiTi", style)
        splash.close()

    def test_main_menu_stays_horizontal_when_font_needs_more_height(self):
        from main_window import MainWindow

        menu_bar = QMenuBar()
        menu_bar.setStyleSheet(
            "QMenuBar { font-size: 30px; }"
            "QMenuBar::item { padding: 2px 8px; }"
        )
        for title in ("功能", "硬件", "用户", "帮助"):
            menu_bar.addMenu(title)

        menu_row = MainWindow._create_menu_row(menu_bar)
        menu_row.resize(1200, menu_row.sizeHint().height())
        menu_row.show()
        self.app.processEvents()

        extension_button = menu_bar.findChild(QToolButton, "qt_menubar_ext_button")
        self.assertGreaterEqual(menu_bar.height(), menu_bar.sizeHint().height())
        self.assertTrue(extension_button is None or not extension_button.isVisible())

        menu_row.close()

    def test_toolbar_removes_visible_condition_mode_control(self):
        toolbar = SequenceToolsBar()

        toolbar_labels = [label.text().strip() for label in toolbar.findChildren(QLabel)]
        self.assertNotIn("模式：", toolbar_labels)
        self.assertTrue(toolbar.condition_mode_combobox.isHidden())
        self.assertEqual(toolbar.condition_mode_combobox.itemText(0), "测试")
        self.assertEqual(toolbar.condition_mode_combobox.itemText(1), "标记")

    def test_toolbar_comboboxes_show_dropdown_arrow(self):
        toolbar = SequenceToolsBar()

        self.assertIn("QComboBox::down-arrow", toolbar.using_file_combobox.styleSheet())
        self.assertIn(ui_style_const.COMBO_DOWN_ARROW_ICON, toolbar.using_file_combobox.styleSheet())

    def test_using_config_combobox_refreshes_before_popup(self):
        toolbar = SequenceToolsBar()
        calls = []
        toolbar.using_file_combobox.before_show_popup = lambda: calls.append("refresh")

        toolbar.using_file_combobox.showPopup()
        self.app.processEvents()
        toolbar.using_file_combobox.hidePopup()

        self.assertEqual(calls, ["refresh"])

    def test_toolbar_has_sample_number_and_current_round_inputs(self):
        toolbar = SequenceToolsBar()

        toolbar.sample_number_lineedit.setText("SAMPLE-001")
        toolbar.current_round_spinbox.setValue(12)

        self.assertEqual(toolbar.sample_number_lineedit.objectName(), "sampleNumberLineEdit")
        self.assertEqual(toolbar.sample_number_lineedit.text(), "SAMPLE-001")
        self.assertEqual(toolbar.sample_number_lineedit.accessibleName(), "样本编号")
        self.assertEqual(toolbar.current_round_spinbox.objectName(), "currentRoundSpinBox")
        self.assertEqual(toolbar.current_round_spinbox.minimum(), 1)
        self.assertEqual(toolbar.current_round_spinbox.maximum(), 9999)
        self.assertEqual(toolbar.current_round_spinbox.value(), 12)
        self.assertEqual(toolbar.current_round_spinbox.buttonSymbols(), QAbstractSpinBox.NoButtons)
        self.assertIn(ui_style_const.COLOR_BORDER_STRONG, toolbar.current_round_spinbox.styleSheet())

    def test_toolbar_fields_follow_compact_operator_order(self):
        toolbar = SequenceToolsBar()
        toolbar.using_file_combobox.addItem("test")
        toolbar.resize(1916, toolbar.sizeHint().height())
        toolbar.show()
        self.app.processEvents()

        sample_number_label = next(
            label
            for label in toolbar.findChildren(QLabel)
            if label.text() == "样本编号："
        )
        model_label = next(
            label
            for label in toolbar.findChildren(QLabel)
            if label.text() == "型 号："
        )
        self.assertEqual(
            sample_number_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Fixed,
        )
        self.assertLessEqual(
            toolbar.sample_number_lineedit.geometry().left()
            - sample_number_label.geometry().right(),
            1,
        )
        self.assertEqual(
            model_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Fixed,
        )
        self.assertLessEqual(
            toolbar.lineedit_type.geometry().left()
            - model_label.geometry().right(),
            1,
        )
        self.assertEqual(toolbar.lineedit_type.width(), 160)
        self.assertLess(
            toolbar.lineedit_type.geometry().right(),
            toolbar.using_file_combobox.geometry().left(),
        )
        self.assertEqual(toolbar.using_file_combobox.width(), 200)
        self.assertLess(
            toolbar.using_file_combobox.geometry().right(),
            toolbar.sample_number_lineedit.geometry().left(),
        )
        self.assertLess(
            toolbar.sample_number_lineedit.geometry().right(),
            toolbar.current_round_spinbox.geometry().left(),
        )
        self.assertLess(
            toolbar.current_round_spinbox.geometry().right(),
            toolbar.lineedit_s_or_n.geometry().left(),
        )
        self.assertGreaterEqual(toolbar.lineedit_s_or_n.width(), 240)
        self.assertEqual(
            toolbar.lineedit_s_or_n.sizePolicy().horizontalPolicy(),
            QSizePolicy.Expanding,
        )
        right_gap = toolbar.width() - toolbar.lineedit_s_or_n.geometry().right() - 1
        self.assertGreaterEqual(right_gap, 0)
        self.assertLessEqual(right_gap, 30)

        toolbar.close()

    def test_serial_status_is_combined_into_configuration_button(self):
        toolbar = SequenceToolsBar()

        self.assertFalse(hasattr(toolbar, "serial_trigger_status_label"))
        self.assertEqual(toolbar.serial_trigger_btn.text(), "未连接")
        self.assertEqual(toolbar.serial_trigger_btn.size().width(), 124)
        self.assertEqual(toolbar.serial_trigger_btn.size().height(), 40)
        self.assertEqual(toolbar.serial_trigger_btn.iconSize().width(), 32)
        self.assertEqual(toolbar.serial_trigger_btn.iconSize().height(), 26)

        host = _DummySerialStatusWidget()
        self.addCleanup(host.close)
        host.serial_trigger_btn = toolbar.serial_trigger_btn
        SequenceWidgetSerialTriggerOpsMixin.on_serial_trigger_status_changed(
            host,
            {"connected": True, "has_response": False, "message": "open"},
        )
        self.assertEqual(toolbar.serial_trigger_btn.text(), "已打开")
        self.assertLessEqual(toolbar.serial_trigger_btn.sizeHint().width(), toolbar.serial_trigger_btn.width())

    def test_serial_status_button_uses_original_connection_states(self):
        widget = _DummySerialStatusWidget()

        cases = [
            (
                {"enabled": False, "error": "denied", "connected": False, "message": "off"},
                "未连接",
                ui_style_const.COLOR_TEXT_MUTED,
            ),
            ({"connected": True, "has_response": False, "message": "open"}, "已打开", ui_style_const.COLOR_PRIMARY_HOVER),
            (
                {"connected": True, "has_response": True, "message": "ok"},
                "已连接",
                ui_style_const.COLOR_OK,
            ),
        ]

        for status, expected_text, expected_color in cases:
            with self.subTest(expected_text=expected_text):
                widget.on_serial_trigger_status_changed(status)
                self.assertEqual(widget.serial_trigger_btn.text(), expected_text)
                self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, widget.serial_trigger_btn.styleSheet())
                self.assertIn(ui_style_const.COLOR_BORDER_STRONG, widget.serial_trigger_btn.styleSheet())
                self.assertIn(expected_color, widget.serial_trigger_btn.styleSheet())
                self.assertIn("串口离散输入触发配置", widget.serial_trigger_btn.toolTip())
                self.assertIn(expected_text, widget.serial_trigger_btn.accessibleName())


if __name__ == "__main__":
    unittest.main()
