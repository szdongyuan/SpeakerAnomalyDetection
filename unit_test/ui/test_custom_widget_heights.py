import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from consts.ui_style_const import scale_size_px
from ui.custom_ui_widget.widgets import (
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    MessageBox,
    PushButton,
    SpinBox,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _current_font(widget):
    font = widget.font
    return font() if callable(font) else font


def _localized_standard_button_texts(message_box):
    return {
        standard_button: message_box.button(standard_button).text()
        for standard_button in (
            QMessageBox.Ok,
            QMessageBox.Cancel,
            QMessageBox.Yes,
            QMessageBox.No,
        )
        if message_box.button(standard_button) is not None
    }


def test_message_box_sync_localizes_yes_no_buttons(qapp):
    message_box = MessageBox()
    message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

    message_box._sync_buttons_style_and_text()

    assert _localized_standard_button_texts(message_box) == {
        QMessageBox.Yes: "确认",
        QMessageBox.No: "取消",
    }
    assert message_box.standardButton(message_box.button(QMessageBox.Yes)) == QMessageBox.Yes
    assert message_box.standardButton(message_box.button(QMessageBox.No)) == QMessageBox.No


def test_message_box_sync_preserves_ok_cancel_button_localization(qapp):
    message_box = MessageBox()
    message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    message_box._sync_buttons_style_and_text()

    assert _localized_standard_button_texts(message_box) == {
        QMessageBox.Ok: "确认",
        QMessageBox.Cancel: "取消",
    }
    assert message_box.standardButton(message_box.button(QMessageBox.Ok)) == QMessageBox.Ok
    assert message_box.standardButton(message_box.button(QMessageBox.Cancel)) == QMessageBox.Cancel


@pytest.mark.parametrize(
    ("factory", "expected_font_px"),
    [
        (LineEdit, scale_size_px(20)),
        (SpinBox, scale_size_px(20)),
        (DoubleSpinBox, scale_size_px(20)),
        (ComboBox, scale_size_px(18)),
        (PushButton, scale_size_px(20)),
        (CheckBox, scale_size_px(20)),
    ],
)
def test_common_controls_have_font_safe_natural_height(qapp, factory, expected_font_px):
    widget = factory("启用") if factory in (PushButton, CheckBox) else factory()
    widget.ensurePolished()

    font = _current_font(widget)
    font_metrics = QFontMetrics(font)
    expected_minimum = font_metrics.height() + scale_size_px(8)

    assert font.pixelSize() == expected_font_px
    assert widget.minimumHeight() >= expected_minimum
    assert widget.sizeHint().height() >= expected_minimum
    assert widget.minimumSizeHint().height() >= expected_minimum


def test_common_control_fixed_height_remains_authoritative(qapp):
    widget = DoubleSpinBox()
    widget.setFixedHeight(23)
    widget.show()
    qapp.processEvents()

    assert widget.minimumHeight() == 23
    assert widget.maximumHeight() == 23
    assert widget.height() == 23


def test_common_control_fixed_size_remains_authoritative(qapp):
    widget = LineEdit()
    widget.setFixedSize(QSize(130, 27))
    widget.show()
    qapp.processEvents()

    assert widget.minimumWidth() == 130
    assert widget.maximumWidth() == 130
    assert widget.minimumHeight() == 27
    assert widget.maximumHeight() == 27
    assert widget.size() == QSize(130, 27)
