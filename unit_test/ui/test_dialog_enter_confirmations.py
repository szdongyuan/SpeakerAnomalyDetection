"""Affirmative Enter dispatch through the real confirmation button paths."""

from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QCheckBox, QDialog, QMessageBox, QPushButton, QWidget

from ui.archive_audio_delete_dialog import ArchiveAudioDeleteDialog
from ui.sequence.motor_mode_switch_panel import MotorModeSwitchPanel
from ui.sequence.sequence_widget_round_reset_ops import SequenceWidgetRoundResetOpsMixin
from ui.shared_queue_save_dialog import SharedQueueSaveDialog
from unit_test.ui.test_shared_queue_save import app, change, editor


@pytest.fixture(params=[(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier)],
                ids=["return", "keypad-enter"])
def enter(request):
    return request.param


def show(dialog, app):
    dialog.show()
    dialog.activateWindow()
    app.processEvents()


def act(dialog, widget, action, enter):
    widget.setFocus()
    assert dialog.focusWidget() is (widget.focusProxy() or widget)
    if action == "mouse":
        QTest.mouseClick(widget, Qt.LeftButton)
    else:
        key, modifiers = enter if action == "enter" else (
            Qt.Key_Space if action == "space" else Qt.Key_Escape, Qt.NoModifier)
        QTest.keyClick(widget, key, modifiers)


def dispose(dialog, app):
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("action", ["enter", "escape", "space", "mouse"])
def test_delete_confirmation_cancel_focus(ui_qapp, enter, missing, action):
    dialog = ArchiveAudioDeleteDialog(dict(wav=0 if missing else 1, raw_csv=0, images=0, analysis_csv=0), 1)
    clicks = []
    dialog.delete_button.clicked.connect(lambda: clicks.append("confirm"))
    dialog.cancel_button.clicked.connect(lambda: clicks.append("cancel"))
    try:
        show(dialog, ui_qapp)
        act(dialog, dialog.cancel_button, action, enter)
        assert dialog.result() == (QDialog.Accepted if action == "enter" else QDialog.Rejected)
        assert clicks == (["confirm"] if action == "enter" else [] if action == "escape" else ["cancel"])
        assert dialog.delete_button.isDefault()
        assert not dialog.cancel_button.isDefault()
    finally:
        dispose(dialog, ui_qapp)


@pytest.mark.parametrize("focus", ["cancel", "details"])
def test_shared_confirmation_enter_from_cancel_or_readonly_details(ui_qapp, enter, focus):
    names = SimpleNamespace(product_name="Product", group_name="Port", condition_name="Condition")
    result = SimpleNamespace(issues=[], references=[SimpleNamespace(display_names=[names])])
    dialog = SharedQueueSaveDialog("queue.json", result)
    clicks = []
    dialog.buttonClicked.connect(clicks.append)
    try:
        show(dialog, ui_qapp)
        act(dialog, dialog.cancel_button if focus == "cancel" else dialog.details, "enter", enter)
        assert dialog.result() == QMessageBox.Ok
        assert clicks == [dialog.save_button]
        assert dialog.defaultButton() is dialog.save_button
        assert dialog.escapeButton() is dialog.cancel_button
        assert dialog.save_button.isDefault() and not dialog.cancel_button.isDefault()
    finally:
        dispose(dialog, ui_qapp)


@pytest.mark.parametrize("action", ["escape", "space", "mouse"])
def test_shared_confirmation_cancel_still_available(ui_qapp, action):
    dialog = SharedQueueSaveDialog("queue.json", SimpleNamespace(issues=[], references=[]))
    try:
        show(dialog, ui_qapp)
        act(dialog, dialog.cancel_button, action, (Qt.Key_Return, Qt.NoModifier))
        assert dialog.result() == QMessageBox.Cancel
    finally:
        dispose(dialog, ui_qapp)


class ResetHost(SequenceWidgetRoundResetOpsMixin, QWidget):
    def __init__(self, retry):
        super().__init__()
        self._round_reset_delete_failed = retry
        self._round_data_records = {}


def schedule_modal_action(callback):
    # Always close a modal after observing it, including when the old behavior
    # leaves it open, so a failing assertion cannot hang the test process.
    def drive():
        dialog = QApplication.activeModalWidget()
        try:
            callback(dialog)
        finally:
            if dialog.isVisible():
                dialog.reject()
    QTimer.singleShot(0, drive)


@pytest.mark.parametrize("checked,retry", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("focus", ["cancel", "checkbox"])
def test_round_reset_enter_preserves_delete_choice(ui_qapp, enter, checked, retry, focus):
    host = ResetHost(retry)
    observed, clicks = [], []

    def drive(dialog):
        checkbox = dialog.findChild(QCheckBox)
        checkbox.setChecked(checked)
        confirm = dialog.findChild(QPushButton, "roundResetConfirmButton")
        cancel = dialog.findChild(QPushButton, "roundResetCancelButton")
        confirm.clicked.connect(lambda: clicks.append("confirm"))
        cancel.clicked.connect(lambda: clicks.append("cancel"))
        observed.append((confirm.text(), checkbox.isEnabled(), confirm.isDefault(), cancel.isDefault()))
        # Retry locks the checkbox; use Cancel focus in that state while
        # still verifying that Enter preserves the required deletion choice.
        widget = cancel if focus == "cancel" or retry else checkbox
        act(dialog, widget, "enter", enter)
        observed.append(checkbox.isChecked())

    try:
        schedule_modal_action(drive)
        assert host._confirm_round_reset("round") is checked
        assert clicks == ["confirm"]
        assert observed == [("删除数据并重置" if checked else "重置", not retry, True, False), checked]
    finally:
        dispose(host, ui_qapp)


@pytest.mark.parametrize("action", ["escape", "space", "mouse"])
def test_round_reset_cancel_still_available(ui_qapp, action):
    host = ResetHost(False)
    try:
        schedule_modal_action(lambda dialog: act(
            dialog, dialog.findChild(QPushButton, "roundResetCancelButton"), action,
            (Qt.Key_Return, Qt.NoModifier)))
        assert host._confirm_round_reset("round") is None
    finally:
        dispose(host, ui_qapp)


class ModeBoard:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    def get_mode_state(self):
        return {"mode": self.mode}

    def on_test_btn_clicked(self):
        self.mode = "test"
        self.calls.append("test")

    def on_mark_btn_clicked(self):
        self.mode = "mark"
        self.calls.append("mark")


@pytest.mark.parametrize("target", ["test", "mark"])
@pytest.mark.parametrize("action", ["enter", "escape", "space", "mouse"])
def test_mode_confirmation_preserves_target_and_cancel(ui_qapp, enter, target, action):
    original = "mark" if target == "test" else "test"
    board = ModeBoard(original)
    panel = MotorModeSwitchPanel(board)
    defaults, clicks = [], []

    def drive(dialog):
        defaults.append((dialog.defaultButton() is dialog.button(QMessageBox.Yes),
                         dialog.button(QMessageBox.Yes).isDefault(),
                         dialog.button(QMessageBox.No).isDefault()))
        dialog.buttonClicked.connect(lambda button: clicks.append(dialog.standardButton(button)))
        act(dialog, dialog.button(QMessageBox.No), action, enter)

    try:
        schedule_modal_action(drive)
        getattr(panel, "_on_" + target + "_clicked")()
        assert board.mode == (target if action == "enter" else original)
        assert board.calls == ([target] if action == "enter" else [])
        assert defaults == [(True, True, False)]
        assert clicks == [QMessageBox.Yes if action == "enter" else QMessageBox.No]
    finally:
        dispose(panel, ui_qapp)


def test_queue_save_held_enter_waits_for_new_press(editor, ui_qapp, enter):
    window, target, registry, products, _ = editor
    before = target.read_bytes(), registry.read_bytes(), (products / "one.json").read_bytes()
    change(window)
    window.confirm_shared_save = window._confirm_shared_save
    show(window, ui_qapp)
    save = next(button for button in window.findChildren(QPushButton) if button.text() == "保存")
    parent_clicks, child_clicks, observed = [], [], []
    save.clicked.connect(lambda: parent_clicks.append("save"))

    def drive(dialog):
        observed.append(isinstance(dialog, SharedQueueSaveDialog))
        dialog.buttonClicked.connect(lambda button: child_clicks.append(dialog.standardButton(button)))
        ui_qapp.sendEvent(dialog, QKeyEvent(QEvent.KeyPress, enter[0], enter[1], "\r", True, 1))
        observed.append((list(child_clicks), dialog.isVisible(), target.read_bytes() == before[0]))
        QTest.keyRelease(dialog, *enter)
        QTest.keyClick(dialog, *enter)

    schedule_modal_action(drive)
    QTest.keyPress(save, *enter)
    assert observed == [True, ([], True, True)]
    assert parent_clicks == ["save"]
    assert child_clicks == [QMessageBox.Ok]
    assert target.read_bytes() != before[0]
    assert registry.read_bytes() == before[1]
    assert (products / "one.json").read_bytes() == before[2]
    assert not window.dirty and not window.isVisible()
