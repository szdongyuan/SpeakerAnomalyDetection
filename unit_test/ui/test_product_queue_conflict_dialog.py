import sys

import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QFrame, QLabel, QMessageBox


@pytest.fixture
def conflict_dialog(ui_qapp):
    from ui.product_queue_conflict_dialog import ProductQueueConflictDialog

    if sys.platform == "win32" and ui_qapp.platformName() == "offscreen":
        pytest.skip("Qt 5 Windows QMessageBox.showEvent crashes with offscreen; run with QT_QPA_PLATFORM=windows")
    original_font = ui_qapp.font()
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    if font_id >= 0:
        ui_qapp.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    windows = []
    def create(rows, parent=None):
        dialog = ProductQueueConflictDialog("采样率、量程不一致：", rows, "无法保存", parent)
        if parent is not None:
            windows.append(parent)
        windows.append(dialog)
        dialog.show()
        dialog.activateWindow()
        ui_qapp.processEvents()
        return dialog
    yield create
    for dialog in reversed(windows):
        dialog.close()
        dialog.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()
    ui_qapp.setFont(original_font)
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


def test_full_readonly_list_between_prompt_and_button(conflict_dialog, ui_qapp):
    rows = tuple(f"端口{n}/档位的测试队列“队列”（采样率 48000 Hz，量程 ±10 V）" for n in range(3))
    dialog = conflict_dialog(rows)
    prompt, listing, buttons = dialog.prompt_label, dialog.conflict_list, dialog.buttons
    assert isinstance(dialog, QMessageBox)
    assert dialog.icon() == QMessageBox.Warning
    icon = dialog.findChild(QLabel, "qt_msgboxex_icon_label")
    assert icon is not None and icon.isVisible() and not icon.pixmap().isNull()
    assert icon.geometry().right() < prompt.geometry().left()
    assert dialog.windowTitle() == "无法保存"
    assert prompt.text() == "采样率、量程不一致："
    assert prompt.textFormat() == Qt.PlainText
    list_top = listing.mapTo(dialog, listing.rect().topLeft())
    assert prompt.geometry().bottom() < list_top.y()
    assert 4 <= list_top.x() - prompt.geometry().left() <= dialog.fontMetrics().height()
    assert list_top.y() + listing.height() <= buttons.geometry().top()
    assert listing.isVisible()
    assert tuple(listing.item(i).text() for i in range(listing.count())) == rows
    assert listing.editTriggers() == QAbstractItemView.NoEditTriggers
    assert all(not listing.item(i).flags() & Qt.ItemIsEditable for i in range(listing.count()))
    assert listing.textElideMode() == Qt.ElideNone and not listing.wordWrap()
    listing.setCurrentRow(1)
    assert [item.text() for item in listing.selectedItems()] == [rows[1]]
    assert buttons.button(QDialogButtonBox.Ok).text() == "确定"
    available = dialog.screen().availableGeometry()
    assert dialog.width() <= available.width() and dialog.height() <= available.height()
    assert listing.frameShape() == QFrame.NoFrame
    assert not listing.viewport().autoFillBackground()
    assert "transparent" in listing.styleSheet() and "border: none" in listing.styleSheet()
    assert dialog.defaultButton() is dialog.button(QMessageBox.Ok)
    assert dialog.escapeButton() is dialog.button(QMessageBox.Ok)


@pytest.mark.parametrize("count", [1, 3, 7])
def test_small_lists_fit_their_content_without_empty_space(conflict_dialog, count):
    dialog = conflict_dialog([f"端口{n}/档位：48000 Hz，±10 V" for n in range(count)])
    listing = dialog.conflict_list
    row_height = listing.sizeHintForRow(0)
    assert count * row_height <= listing.height() <= count * row_height + 2
    assert listing.verticalScrollBar().maximum() == 0
    assert listing.horizontalScrollBar().maximum() == 0
    assert dialog.height() <= listing.height() + 8 * dialog.fontMetrics().height()
    assert dialog.width() < min(600, dialog.screen().availableGeometry().width())


def test_many_long_rows_scroll_without_losing_text(conflict_dialog, ui_qapp):
    rows = tuple(f"端口{n}/" + "超长档位名称" * 40 for n in range(35))
    dialog = conflict_dialog(rows)
    listing = dialog.conflict_list
    assert listing.verticalScrollBar().maximum() > 0
    assert listing.horizontalScrollBar().maximum() > 0
    assert listing.viewport().height() <= listing.sizeHintForRow(0) * 8 + 2
    assert dialog.width() <= dialog.screen().availableGeometry().width()
    assert dialog.height() <= dialog.screen().availableGeometry().height()
    listing.scrollToItem(listing.item(34))
    ui_qapp.processEvents()
    assert listing.item(34).text() == rows[34]
    assert listing.visualItemRect(listing.item(34)).intersects(listing.viewport().rect())


@pytest.mark.parametrize("dismiss", ["button", "return", "enter", "escape", "close"])
def test_dismissal_only_closes_conflict(conflict_dialog, ui_qapp, dismiss):
    parent = QDialog()
    parent.show()
    accepted = []
    parent.accepted.connect(lambda: accepted.append(True))
    dialog = conflict_dialog(["端口/档位的测试队列“队列”（采样率 48000 Hz，量程 ±10 V）"], parent)
    dialog.conflict_list.setFocus()
    if dismiss == "button":
        dialog.buttons.button(QDialogButtonBox.Ok).click()
    elif dismiss == "close":
        dialog.close()
    else:
        key = {"return": Qt.Key_Return, "enter": Qt.Key_Enter, "escape": Qt.Key_Escape}[dismiss]
        QTest.keyClick(dialog.conflict_list, key)
    ui_qapp.processEvents()
    assert not dialog.isVisible()
    assert dialog.result() == QMessageBox.Ok
    assert parent.isVisible() and not accepted
    parent.close()
