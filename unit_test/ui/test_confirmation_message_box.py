"""Real Qt checks for localized confirmation decisions and safe defaults."""

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMessageBox

from ui.confirmation_message_box import ConfirmationMessageBox
from ui.custom_ui_widget.widgets import MessageBox
from ui.config_dialog_base import ConfigDialogBase
from ui.shared_queue_save_dialog import SharedQueueSaveDialog
from types import SimpleNamespace


@pytest.fixture(params=[False, True], ids=["standalone", "themed-parent"])
def parent(qt_app, request):
    widget = ConfigDialogBase() if request.param else None
    yield widget
    if widget is not None:
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("action", ["confirm", "cancel", "enter", "keypad", "escape", "close"])
def test_confirmation_layout_and_decision(qt_app, parent, action):
    dialog = ConfirmationMessageBox(parent, "确认操作", "操作会修改当前配置。",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    dialog.show()
    qt_app.processEvents()
    confirm = dialog.button(QMessageBox.Yes)
    cancel = dialog.button(QMessageBox.No)
    try:
        assert confirm.text() == "确认"
        assert cancel.text() == "取消"
        assert cancel.mapTo(dialog, QPoint()).x() < confirm.mapTo(dialog, QPoint()).x()
        assert dialog.width() - confirm.mapTo(dialog, QPoint()).x() - confirm.width() <= 25
        assert dialog.defaultButton() is cancel
        assert dialog.escapeButton() is cancel
        assert cancel.hasFocus()

        def decide():
            if action in ("confirm", "cancel"):
                QTest.mouseClick(confirm if action == "confirm" else cancel, Qt.LeftButton)
            elif action == "close":
                dialog.close()
            else:
                key = {"enter": Qt.Key_Return, "keypad": Qt.Key_Enter, "escape": Qt.Key_Escape}[action]
                QTest.keyClick(dialog, key)

        QTimer.singleShot(0, decide)
        assert dialog.exec_() == (QMessageBox.Yes if action == "confirm" else QMessageBox.No)
    finally:
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("choice", [QMessageBox.Save, QMessageBox.Discard, QMessageBox.Cancel])
def test_unsaved_queue_three_choices_keep_distinct_results(qt_app, parent, choice):
    dialog = ConfirmationMessageBox(parent, "未保存的测试队列", "是否保存？",
                                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                    QMessageBox.Cancel)
    dialog.show()
    qt_app.processEvents()
    try:
        for standard, label in ((QMessageBox.Save, "保存"), (QMessageBox.Discard, "不保存"),
                                (QMessageBox.Cancel, "取消")):
            assert dialog.button(standard).text() == label
        assert dialog.defaultButton() is dialog.button(QMessageBox.Cancel)
        assert dialog.escapeButton() is dialog.button(QMessageBox.Cancel)
        positions = [dialog.button(standard).mapTo(dialog, QPoint()).x()
                     for standard in (QMessageBox.Discard, QMessageBox.Cancel, QMessageBox.Save)]
        assert positions == sorted(positions)
        QTimer.singleShot(0, lambda: QTest.mouseClick(dialog.button(choice), Qt.LeftButton))
        assert dialog.exec_() == choice
    finally:
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("positive,negative", [(QMessageBox.Yes, QMessageBox.No),
                                               (QMessageBox.Ok, QMessageBox.Cancel)])
def test_shared_message_box_puts_cancel_before_confirm(qt_app, parent, positive, negative):
    dialog = MessageBox(parent)
    dialog.setStandardButtons(positive | negative)
    dialog._sync_buttons_style_and_text()
    dialog.show()
    qt_app.processEvents()
    try:
        assert dialog.button(negative).mapTo(dialog, QPoint()).x() < dialog.button(positive).mapTo(dialog, QPoint()).x()
        assert dialog.standardButton(dialog.button(positive)) == positive
        assert dialog.standardButton(dialog.button(negative)) == negative
    finally:
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_shared_queue_button_order_with_parent_theme(qt_app, parent):
    result = SimpleNamespace(issues=[], references=[])
    dialog = SharedQueueSaveDialog("Q.json", result, parent)
    dialog.show()
    qt_app.processEvents()
    try:
        assert dialog.cancel_button.mapTo(dialog, QPoint()).x() < dialog.save_button.mapTo(dialog, QPoint()).x()
        assert dialog.defaultButton() is dialog.save_button
        assert dialog.escapeButton() is dialog.cancel_button
    finally:
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
