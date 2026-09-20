import sys

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtGui import QInputMethodEvent, QKeyEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDateTimeEdit, QDialog, QDoubleSpinBox,
    QFileDialog, QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout,
)

from ui.dialog_enter_policy import install_dialog_enter_policy


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_editor_is_protected_and_auxiliary_focus_confirms(ui_qapp, key):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    editor = QLineEdit()
    auxiliary = QPushButton("浏览")
    confirm = QPushButton("确定")
    for widget in (editor, auxiliary, confirm):
        layout.addWidget(widget)
    clicks = []
    auxiliary.clicked.connect(lambda: clicks.append("auxiliary"))
    confirm.clicked.connect(lambda: clicks.append("confirm"))
    policy = install_dialog_enter_policy(dialog, confirm)
    assert policy is install_dialog_enter_policy(dialog, confirm)
    dialog.show()
    dialog.activateWindow()
    ui_qapp.processEvents()
    try:
        editor.setFocus()
        QTest.keyClick(editor, key)
        assert clicks == []
        auxiliary.setFocus()
        QTest.keyClick(auxiliary, key)
        assert clicks == ["confirm"]
        confirm.setEnabled(False)
        QTest.keyClick(auxiliary, key)
        assert clicks == ["confirm"]
    finally:
        dialog.close()
        dialog.deleteLater()
        ui_qapp.processEvents()

@pytest.fixture
def scene(ui_qapp):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    auxiliary = QPushButton("Auxiliary")
    confirm = QPushButton("Confirm")
    layout.addWidget(auxiliary)
    layout.addWidget(confirm)
    clicks = []
    auxiliary.clicked.connect(lambda: clicks.append("auxiliary"))
    confirm.clicked.connect(lambda: clicks.append("confirm"))
    policy = install_dialog_enter_policy(dialog, confirm)
    dialog.show()
    dialog.activateWindow()
    ui_qapp.processEvents()
    yield dialog, layout, auxiliary, confirm, clicks, policy
    dialog.close()
    dialog.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
@pytest.mark.parametrize("kind", ["readonly", "spin", "double", "date", "combo", "editable_combo", "text", "plain"])
def test_native_editing_is_preserved(scene, ui_qapp, key, kind):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    if kind == "readonly":
        editor = QLineEdit("C:/recordings")
        editor.setReadOnly(True)
    elif kind in ("spin", "double"):
        editor = QSpinBox() if kind == "spin" else QDoubleSpinBox()
        editor.setKeyboardTracking(False)
    elif kind == "date":
        editor = QDateTimeEdit()
    elif "combo" in kind:
        editor = QComboBox()
        editor.addItems(["first", "second"])
        editor.setEditable(kind == "editable_combo")
    else:
        editor = QTextEdit() if kind == "text" else QPlainTextEdit()
    layout.insertWidget(0, editor)
    ui_qapp.processEvents()
    editor.setFocus()
    completed = []
    if hasattr(editor, "editingFinished"):
        editor.editingFinished.connect(lambda: completed.append(True))
    receiver = editor.findChild(QLineEdit) if kind in ("spin", "double", "date", "editable_combo") else editor
    if kind in ("spin", "double"):
        receiver.setText("42")
    QTest.keyClick(receiver, key)
    assert clicks == []
    if kind in ("spin", "double"):
        assert editor.value() == 42
    if kind in ("readonly", "spin", "double", "date"):
        assert completed
    if kind in ("text", "plain"):
        assert editor.toPlainText() == "\n"
    if kind == "readonly":
        assert editor.text() == "C:/recordings"


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_combo_popup_selects_without_confirming(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QComboBox()
    editor.addItems(["first", "second"])
    layout.addWidget(editor)
    editor.setFocus()
    editor.showPopup()
    ui_qapp.processEvents()
    QTest.keyClick(editor.view(), Qt.Key_Down)
    QTest.keyClick(editor.view(), key)
    assert editor.currentText() == "second"
    assert clicks == []
    assert not editor.view().isVisible()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_dynamic_cell_editor_commits_without_leaking_to_dialog(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    table = QTableWidget(1, 1)
    table.setItem(0, 0, QTableWidgetItem("old"))
    layout.addWidget(table)
    ui_qapp.processEvents()
    table.editItem(table.item(0, 0))
    editor = table.findChild(QLineEdit)
    assert editor is not None
    editor.setText("new")
    editor.editingFinished.connect(auxiliary.setFocus)
    QTest.keyClick(editor, key)
    ui_qapp.processEvents()
    assert table.item(0, 0).text() == "new"
    assert clicks == []
    QTest.keyClick(table, key)
    assert clicks == ["confirm"]


@pytest.mark.parametrize("kind", [QTextEdit, QPlainTextEdit])
def test_readonly_explanation_is_non_input(scene, ui_qapp, kind):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    explanation = kind("Explanation")
    explanation.setReadOnly(True)
    layout.addWidget(explanation)
    explanation.setFocus()
    QTest.keyClick(explanation, Qt.Key_Return)
    assert clicks == ["confirm"]


def test_input_remains_protected_when_editing_finished_moves_focus(scene):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit()
    layout.addWidget(editor)
    editor.editingFinished.connect(auxiliary.setFocus)
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(auxiliary, Qt.Key_Return)
    assert clicks == ["confirm"]

@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_target_updates_and_default_visual_role(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    assert confirm.isDefault()
    assert not auxiliary.autoDefault()
    policy.set_confirm_button(auxiliary)
    assert auxiliary.isDefault()
    assert not confirm.isDefault()
    QTest.keyClick(confirm, key)
    assert clicks == ["auxiliary"]
    assert policy is install_dialog_enter_policy(dialog)
    assert not auxiliary.isDefault()
    for button in (auxiliary, confirm):
        button.setFocus()
        QTest.keyClick(button, key)
    assert clicks == ["auxiliary"]


def test_hidden_target_and_dynamic_default_button_do_not_fall_back(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    late = QPushButton("Late")
    late.setDefault(True)
    layout.addWidget(late)
    late.clicked.connect(lambda: clicks.append("late"))
    ui_qapp.processEvents()
    assert not late.autoDefault()
    assert not late.isDefault()
    confirm.hide()
    QTest.keyClick(late, Qt.Key_Return)
    assert clicks == []
    confirm.show()
    dialog.hide()
    dialog.show()
    QTest.keyClick(late, Qt.Key_Return)
    assert clicks == ["confirm"]


def test_deleted_confirmation_target_is_safe(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    confirm.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(confirm)
    QTest.keyClick(auxiliary, Qt.Key_Return)
    assert clicks == []


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_ime_preedit_does_not_confirm_window(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit()
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    assert dialog.focusWidget() is editor
    ui_qapp.sendEvent(editor, QInputMethodEvent("zhong", []))
    QTest.keyClick(editor, key)
    assert clicks == []
    QTest.keyClick(dialog, key)
    assert clicks == []
    commit = QInputMethodEvent()
    commit.setCommitString("中")
    ui_qapp.sendEvent(editor, commit)
    assert editor.text() == "中"
    auxiliary.setFocus()
    QTest.keyClick(auxiliary, key)
    assert clicks == ["confirm"]


def test_active_menu_interaction_does_not_confirm_parent(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    menu = QMenu(dialog)
    action = menu.addAction("Choose")
    selected = []
    action.triggered.connect(lambda: selected.append(True))
    menu.popup(dialog.mapToGlobal(dialog.rect().center()))
    menu.setActiveAction(action)
    ui_qapp.processEvents()
    QTest.keyClick(menu, Qt.Key_Return)
    assert selected == [True]
    assert clicks == []
    menu.close()


@pytest.mark.parametrize("key, modifiers", [
    (Qt.Key_Return, Qt.NoModifier),
    (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_repeat_cannot_confirm_new_modal_dialog(scene, ui_qapp, key, modifiers):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    if sys.platform == "win32" and ui_qapp.platformName() == "offscreen":
        pytest.skip("Windows Qt QMessageBox crashes offscreen; run this test with QT_QPA_PLATFORM=windows")
    child = QMessageBox(QMessageBox.Question, "Question", "Proceed?", QMessageBox.Yes | QMessageBox.No, dialog)
    child.setDefaultButton(QMessageBox.Yes)
    child_clicks = []
    child.buttonClicked.connect(lambda button: child_clicks.append(button))
    observed = []

    def exercise_child():
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyPress, key, modifiers, "\r", True, 1))
        observed.append(list(child_clicks))
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyRelease, key, modifiers))
        QTest.keyClick(child, key, modifiers)
        if child.isVisible():
            child.reject()

    def open_child():
        QTimer.singleShot(0, exercise_child)
        child.exec()

    confirm.clicked.connect(open_child)
    QTest.keyClick(auxiliary, key, modifiers)
    assert observed == [[]]
    assert len(child_clicks) == 1
    assert clicks == ["confirm"]
    child.deleteLater()


def test_two_dialogs_and_modal_boundary_are_isolated(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    child = QDialog(dialog)
    child_layout = QVBoxLayout(child)
    child_confirm = QPushButton("Child confirm")
    child_layout.addWidget(child_confirm)
    child_clicks = []
    child_confirm.clicked.connect(lambda: child_clicks.append(True))
    install_dialog_enter_policy(child, child_confirm)
    child.setModal(True)
    child.show()
    child.activateWindow()
    ui_qapp.processEvents()
    QTest.keyClick(auxiliary, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(child_confirm, Qt.Key_Return)
    assert child_clicks == [True]
    assert clicks == []
    child.close()
    child.deleteLater()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_repeat_does_not_repeat_confirmation(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    QTest.keyPress(auxiliary, key)
    for _ in range(3):
        ui_qapp.sendEvent(auxiliary, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "\r", True, 1))
    QTest.keyRelease(auxiliary, key)
    assert clicks == ["confirm"]
    QTest.keyClick(auxiliary, key)
    assert clicks == ["confirm", "confirm"]


def test_existing_mouse_space_tab_escape_and_modifiers(scene):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    QTest.mouseClick(auxiliary, Qt.LeftButton)
    auxiliary.setFocus()
    QTest.keyClick(auxiliary, Qt.Key_Space)
    assert clicks == ["auxiliary", "auxiliary"]
    QTest.keyClick(auxiliary, Qt.Key_Tab)
    assert dialog.focusWidget() is confirm
    for modifier in (Qt.ControlModifier, Qt.AltModifier, Qt.ShiftModifier):
        QTest.keyClick(auxiliary, Qt.Key_Return, modifier)
    assert clicks == ["auxiliary", "auxiliary"]
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert dialog.result() == QDialog.Rejected
    assert not dialog.isVisible()


def test_uninstalled_dialog_preserves_native_default(ui_qapp):
    from ui.ui_analysis_config.common_widgets import ConfigDialogBase

    for dialog in (QDialog(), ConfigDialogBase()):
        layout = dialog.layout() or QVBoxLayout(dialog)
        button = QPushButton("Native default")
        layout.addWidget(button)
        clicks = []
        button.clicked.connect(lambda: clicks.append(True))
        dialog.show()
        dialog.activateWindow()
        ui_qapp.processEvents()
        button.setAutoDefault(True)
        button.setDefault(True)
        button.setFocus()
        QTest.keyClick(button, Qt.Key_Return)
        assert clicks == [True]
        assert not hasattr(dialog, "_enter_policy")
        dialog.close()
        dialog.deleteLater()

@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_dialog_delivery_respects_current_parameter_focus(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit("value")
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    assert dialog.focusWidget() is editor
    QTest.keyClick(dialog, key)
    assert clicks == []


def test_destroyed_policy_is_owned_by_dialog(ui_qapp):
    dialog = QDialog()
    policy = install_dialog_enter_policy(dialog)
    assert policy.parent() is dialog
    dialog.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(policy)


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_no_target_and_non_input_controls(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    checkbox = QCheckBox("Option")
    layout.addWidget(checkbox)
    ui_qapp.processEvents()
    checkbox.setFocus()
    QTest.keyClick(checkbox, key)
    assert not checkbox.isChecked()
    assert clicks == ["confirm"]
    policy.set_confirm_button(None)
    for widget in (dialog, auxiliary, confirm, checkbox):
        widget.setFocus()
        QTest.keyClick(widget, key)
    assert clicks == ["confirm"]


def test_independent_unrelated_dialog_keeps_its_own_target(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    other = QDialog()
    other_layout = QVBoxLayout(other)
    target = QPushButton("Other")
    other_layout.addWidget(target)
    other_clicks = []
    target.clicked.connect(lambda: other_clicks.append(True))
    install_dialog_enter_policy(other, target)
    other.show()
    other.activateWindow()
    ui_qapp.processEvents()
    QTest.keyClick(target, Qt.Key_Return)
    assert other_clicks == [True]
    assert clicks == []
    other.close()
    other.deleteLater()


@pytest.mark.parametrize("native", [False, True])
def test_file_dialog_modal_blocks_parent_confirmation(scene, ui_qapp, native):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    if native and ui_qapp.platformName() != "windows":
        pytest.skip("Native Windows file chooser requires QT_QPA_PLATFORM=windows")
    child = QFileDialog(dialog)
    # No accept operation or filesystem mutation: only modal ownership is tested.
    child.setOption(QFileDialog.DontUseNativeDialog, not native)
    child.setModal(True)
    child.show()
    ui_qapp.processEvents()
    QTest.keyClick(auxiliary, Qt.Key_Return)
    assert clicks == []
    child.reject()
    child.deleteLater()


def test_editor_release_in_child_window_does_not_poison_next_press(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit()
    layout.addWidget(editor)
    child = QDialog(dialog)

    def finish_editing():
        def release_and_close():
            QTest.keyRelease(child, Qt.Key_Return)
            child.reject()
        QTimer.singleShot(0, release_and_close)
        child.exec()
        auxiliary.setFocus()

    editor.returnPressed.connect(finish_editing)
    editor.setFocus()
    QTest.keyPress(editor, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(auxiliary, Qt.Key_Return)
    assert clicks == ["confirm"]
    child.deleteLater()


@pytest.mark.parametrize("key, modifiers", [
    (Qt.Key_Return, Qt.NoModifier),
    (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Enter, Qt.KeypadModifier),
])
@pytest.mark.parametrize("signal_name", ["returnPressed", "editingFinished"])
def test_editor_callback_modal_rejects_held_repeat_until_release(scene, ui_qapp, key, modifiers, signal_name):
    if sys.platform == "win32" and ui_qapp.platformName() == "offscreen":
        pytest.skip("Windows Qt QMessageBox crashes offscreen; run this test with QT_QPA_PLATFORM=windows")
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit("value")
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    child = QMessageBox(QMessageBox.Question, "Question", "Proceed?", QMessageBox.Yes | QMessageBox.No, dialog)
    child.setDefaultButton(QMessageBox.Yes)
    child_clicks = []
    child.buttonClicked.connect(lambda button: child_clicks.append(button))
    observed = []

    def exercise_child():
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyPress, key, modifiers, "\r", True, 1))
        observed.append(list(child_clicks))
        QTest.keyRelease(child, key, modifiers)
        QTest.keyClick(child, key, modifiers)
        if child.isVisible():
            child.reject()

    def open_child():
        # Test one callback, not the additional focus-out notification caused
        # by displaying a modal dialog from an editingFinished handler.
        getattr(editor, signal_name).disconnect(open_child)
        QTimer.singleShot(0, exercise_child)
        child.exec()

    getattr(editor, signal_name).connect(open_child)
    QTest.keyClick(editor, key, modifiers)
    assert observed == [[]]
    assert child_clicks == [child.button(QMessageBox.Yes)]
    assert clicks == []
    child.deleteLater()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_native_multiline_repeat_remains_available(scene, ui_qapp, key):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QPlainTextEdit()
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    QTest.keyPress(editor, key)
    ui_qapp.sendEvent(editor, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "\r", True, 1))
    QTest.keyRelease(editor, key)
    assert editor.toPlainText() == "\n\n"
    assert clicks == []



def test_keypad_origin_protects_editor_and_routes_auxiliary_confirmation(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    editor = QLineEdit("value")
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == []
    auxiliary.setFocus()
    QTest.keyClick(auxiliary, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ["confirm"]
    policy.set_confirm_button(None)
    for widget in (editor, auxiliary, confirm):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ["confirm"]


def test_keypad_origin_repeat_does_not_repeat_confirmation(scene, ui_qapp):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    auxiliary.setFocus()
    QTest.keyPress(auxiliary, Qt.Key_Enter, Qt.KeypadModifier)
    ui_qapp.sendEvent(auxiliary, QKeyEvent(QEvent.KeyPress, Qt.Key_Enter, Qt.KeypadModifier, "\r", True, 1))
    QTest.keyRelease(auxiliary, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ["confirm"]
    QTest.keyClick(auxiliary, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ["confirm", "confirm"]


@pytest.mark.parametrize("modifier", [Qt.ControlModifier, Qt.AltModifier, Qt.ShiftModifier, Qt.MetaModifier])
def test_keypad_with_command_modifier_preserves_custom_key_handler(scene, ui_qapp, modifier):
    dialog, layout, auxiliary, confirm, clicks, policy = scene
    received = []

    class ShortcutButton(QPushButton):
        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Enter:
                received.append(event.modifiers())
                event.accept()
            else:
                super().keyPressEvent(event)

    button = ShortcutButton("Existing modified-Enter handler")
    layout.addWidget(button)
    ui_qapp.processEvents()
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Enter, modifier | Qt.KeypadModifier)
    assert received == [modifier | Qt.KeypadModifier]
    assert clicks == []
