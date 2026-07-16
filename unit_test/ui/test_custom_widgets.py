import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

from ui.custom_ui_widget.widgets import TableWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def assert_table_fonts_match(table):
    body_font = QWidget.font(table)
    for header in (
        table.horizontalHeader(),
        table.verticalHeader(),
    ):
        header_font = header.font()
        assert header.testAttribute(Qt.WA_SetFont)
        assert header_font.family() == body_font.family()
        assert header_font.pixelSize() == body_font.pixelSize()


def test_table_widget_applies_font_to_headers_on_construction(qapp):
    table = TableWidget()
    assert_table_fonts_match(table)


def test_table_widget_set_font_size_updates_headers(qapp):
    table = TableWidget()
    table.set_font_size(37)

    assert QWidget.font(table).pixelSize() == table.font_size
    assert_table_fonts_match(table)
