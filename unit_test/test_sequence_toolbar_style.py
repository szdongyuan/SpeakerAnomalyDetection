import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QFont, QStandardItem
from PyQt5.QtTest import QTest

from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QLabel,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from ui.sequence.sequence_tools_bar import SequenceToolsBar

class TestSequenceToolbarStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_toolbar_buttons_keep_accessibility_text(self):
        toolbar = SequenceToolsBar()
        expectations = [
            (toolbar.player_btn, "开始录制"),
        ]

        for button, expected in expectations:
            with self.subTest(expected=expected):
                self.assertEqual(button.toolTip(), expected)
                self.assertEqual(button.accessibleName(), expected)
                self.assertEqual(button.accessibleDescription(), expected)
                self.assertIn(ui_style_const.COLOR_TOOLBAR_BUTTON_BG, button.styleSheet())

    def test_product_fields_follow_action_buttons_before_flexible_space(self):
        toolbar = SequenceToolsBar()
        toolbar.resize(1920, toolbar.sizeHint().height())
        toolbar.show()
        self.app.processEvents()
        self.assertLess(toolbar.player_btn.geometry().right(),
                        toolbar.lineedit_type.geometry().left())
        toolbar.close()

    def test_toolbar_uses_light_blue_container_style(self):
        toolbar = SequenceToolsBar()

        self.assertEqual(toolbar.objectName(), "sequenceToolsBar")
        self.assertIn(ui_style_const.COLOR_TOOLBAR_BG, toolbar.styleSheet())

    def assert_font(self, widget, size, *, family=ui_style_const.MAIN_UI_SMALL_FONT_FAMILY_NAME):
        widget.ensurePolished()
        font = widget.font()
        self.assertEqual((font.family(), font.pixelSize(), font.weight()), (family, size, QFont.Normal))

    def test_main_window_controls_menu_and_tooltip_fonts_are_local(self):
        import main_window as module

        with (
            patch.object(module.LogManager, "set_log_handler", logging.getLogger),
            patch.object(module, "restore_or_default", return_value=(None, None, [], [])),
            patch.object(module.MainWindow, "init_ui"),
            patch.object(module.MainWindow, "_init_ve_hardware_runtime"),
            patch.object(module.MainWindow, "get_current_version", return_value="test"),
        ):
            window = module.MainWindow(
                recording_bridge=SimpleNamespace(shutdown=lambda: None),
                raw_audio_csv_bridge=SimpleNamespace(service=None),
            )
            title = window.set_title()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.hide)
        center = QWidget()
        layout = QVBoxLayout(center)
        window.setCentralWidget(center)
        layout.addWidget(title)
        menu_bar = window.init_menu()
        layout.addWidget(window._create_menu_row(menu_bar))
        toolbar = SequenceToolsBar()
        layout.addWidget(toolbar)
        window.show_statusbar_layout()
        window.show()
        self.app.processEvents()

        title_label = next(label for label in title.findChildren(QLabel) if label.text())
        self.assert_font(title_label, 20)
        self.assert_font(menu_bar, 18)
        menu = menu_bar.actions()[0].menu()
        menu.popup(menu_bar.mapToGlobal(QPoint(0, menu_bar.height())))
        try:
            self.app.processEvents()
            self.assertTrue(menu.isVisible())
            self.assert_font(menu, 18)
        finally:
            menu.close()
        self.assert_font(window.user_label, 15)
        self.assert_font(window.device_label, 15)
        self.assert_font(toolbar.lineedit_type, 18)
        self.assert_font(toolbar.sample_number_lineedit, 18)
        self.assert_font(toolbar.lineedit_s_or_n, 18)
        self.assert_font(toolbar.using_file_combobox, 18)
        self.assert_font(toolbar.barcode_scanner_box, 18)
        for label in toolbar.findChildren(QLabel):
            if label.buddy() is not None:
                self.assert_font(label, 18)

        dialog = QDialog(window)
        layout = QVBoxLayout(dialog)
        label = QLabel("Independent label")
        button = QPushButton("Independent button")
        button.setToolTip("Independent hint")
        layout.addWidget(label)
        layout.addWidget(button)
        self.addCleanup(dialog.close)
        self.assert_font(label, 20, family="SimSun")
        self.assert_font(button, 20, family="SimSun")
        tooltip_font = QToolTip.font()
        for widget, expected in (
            (toolbar.player_btn, (ui_style_const.MAIN_UI_SMALL_FONT_FAMILY_NAME, 16)),
            (button, (tooltip_font.family(), tooltip_font.pixelSize())),
        ):
            if widget is button:
                dialog.show()
                self.app.processEvents()
            try:
                QToolTip.showText(widget.mapToGlobal(QPoint(1, 1)), widget.toolTip(), widget)
                QTest.qWait(30)
                tip = next(w for w in self.app.allWidgets() if w.objectName() == "qtooltip_label" and w.isVisible())
                self.assertEqual((tip.font().family(), tip.font().pixelSize()), expected)
            finally:
                QToolTip.hideText()
                QTest.qWait(350)

    def test_independent_editor_context_menu_keeps_simsun(self):
        from ui.operation_sequence import OptionList

        with patch.object(OptionList, "load_model_config"):
            options = OptionList(logging.getLogger(), "")
        self.addCleanup(options.close)
        options.model().appendRow(QStandardItem("Example"))
        options.show()
        self.app.processEvents()
        seen = []

        def inspect_menu(menu, pos):
            menu.popup(pos)
            try:
                self.app.processEvents()
                self.assertTrue(menu.isVisible())
                seen.append((menu.font().family(), menu.font().pixelSize()))
            finally:
                menu.close()

        with patch.object(QMenu, "exec_", inspect_menu):
            options.show_context_menu(options.visualRect(options.model().index(0, 0)).center())
        self.assertEqual(seen, [("SimSun", 18)])
        self.assertIn(ui_style_const.UI_FONT_FAMILY, ui_style_const.toolbar_button_style)

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

    def test_splash_uses_the_shared_ui_font(self):
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
        self.assertFalse(hasattr(toolbar, "condition_mode_combobox"))

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

    def test_toolbar_shows_only_read_only_daily_round(self):
        toolbar = SequenceToolsBar()
        toolbar.sample_number_lineedit.setText("10000")
        self.assertEqual(toolbar.sample_number_lineedit.text(), "10000")
        self.assertTrue(toolbar.sample_number_lineedit.isReadOnly())
        self.assertEqual(toolbar.sample_number_lineedit.accessibleName(), "轮次")
        self.assertFalse(hasattr(toolbar, "current_round_spinbox"))

    def test_toolbar_fields_follow_compact_operator_order(self):
        toolbar = SequenceToolsBar()
        toolbar.using_file_combobox.addItem("test")
        toolbar.resize(1916, toolbar.sizeHint().height())
        toolbar.show()
        self.app.processEvents()

        sample_number_label = next(
            label
            for label in toolbar.findChildren(QLabel)
            if label.buddy() is toolbar.sample_number_lineedit
        )
        model_label = next(
            label
            for label in toolbar.findChildren(QLabel)
            if label.buddy() is toolbar.lineedit_type
        )
        self.assertEqual(
            sample_number_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Preferred,
        )
        self.assertLessEqual(
            toolbar.sample_number_lineedit.geometry().left()
            - sample_number_label.geometry().right(),
            5,
        )
        self.assertEqual(
            model_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Preferred,
        )
        self.assertLessEqual(
            toolbar.lineedit_type.geometry().left()
            - model_label.geometry().right(),
            5,
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
            toolbar.lineedit_s_or_n.geometry().left(),
        )
        self.assertEqual(toolbar.lineedit_s_or_n.width(), 320)
        self.assertEqual(
            toolbar.lineedit_s_or_n.sizePolicy().horizontalPolicy(),
            QSizePolicy.Expanding,
        )
        right_gap = toolbar.width() - toolbar.lineedit_s_or_n.geometry().right() - 1
        self.assertGreaterEqual(right_gap, 0)
        self.assertGreater(right_gap, 30)

        toolbar.close()

if __name__ == "__main__":
    unittest.main()
