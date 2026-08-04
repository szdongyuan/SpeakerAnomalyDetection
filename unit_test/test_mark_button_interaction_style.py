import unittest

from consts import ui_style_const


class TestMarkButtonInteractionStyle(unittest.TestCase):
    def test_count_board_ok_ng_buttons_have_hover_and_pressed_feedback(self):
        self.assertIn("QPushButton:hover", ui_style_const.count_board_ok_button_style)
        self.assertIn("border-color: #15803D", ui_style_const.count_board_ok_button_style)
        self.assertIn("background-color: #BCE8CF", ui_style_const.count_board_ok_button_style)
        self.assertIn("padding: 7px 10px 3px 14px", ui_style_const.count_board_ok_button_style)

        self.assertIn("QPushButton:hover", ui_style_const.count_board_ng_button_style)
        self.assertIn("border-color: #B91C1C", ui_style_const.count_board_ng_button_style)
        self.assertIn("background-color: #F4B7B7", ui_style_const.count_board_ng_button_style)
        self.assertIn("padding: 7px 10px 3px 14px", ui_style_const.count_board_ng_button_style)

    def test_waveform_mark_buttons_have_status_specific_hover_and_pressed_feedback(self):
        style = ui_style_const.waveform_mark_button_style

        self.assertIn("QPushButton#conditionOkButton:hover", style)
        self.assertIn("QPushButton#conditionOkButton:pressed", style)
        self.assertIn("background-color: #C8F1D8", style)
        self.assertIn("background-color: #A9E2C1", style)
        self.assertIn("border-width: 2px", style)

        self.assertIn("QPushButton#conditionNgButton:hover", style)
        self.assertIn("QPushButton#conditionNgButton:pressed", style)
        self.assertIn("background-color: #F8BEBE", style)
        self.assertIn("background-color: #EF9F9F", style)

        self.assertIn("padding-top: 2px", style)
        self.assertIn("padding-left: 1px", style)


if __name__ == "__main__":
    unittest.main()
