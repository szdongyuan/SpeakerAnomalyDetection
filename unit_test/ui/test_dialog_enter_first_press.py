import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from PyQt5.QtCore import QDateTime, QEvent, QTimer, Qt
from PyQt5.QtGui import QInputMethodEvent, QIntValidator, QKeyEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDateTimeEdit, QDialog, QLineEdit, QMenu, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
)

from ui.dialog_enter_policy import install_dialog_enter_policy


@pytest.fixture
def scene(ui_qapp):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    first, second = QLineEdit('first'), QLineEdit('second')
    confirm, auxiliary = QPushButton('Confirm'), QPushButton('Auxiliary')
    for widget in (first, second, auxiliary, confirm):
        layout.addWidget(widget)
    clicks = []
    confirm.clicked.connect(lambda: clicks.append('confirm'))
    auxiliary.clicked.connect(lambda: clicks.append('auxiliary'))
    policy = install_dialog_enter_policy(dialog, confirm, confirm_on_first_enter=True)
    dialog.show()
    dialog.activateWindow()
    ui_qapp.processEvents()
    first.setFocus()
    yield dialog, first, second, confirm, auxiliary, clicks, policy
    dialog.close()
    dialog.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()


@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
@pytest.mark.parametrize('delivery', ['editor', 'dialog'])
def test_first_enter_confirms_once_without_native_enter(scene, ui_qapp, key, modifiers, delivery):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    first.setFocus()
    ui_qapp.processEvents()
    signals = []
    first.returnPressed.connect(lambda: signals.append('return'))
    first.editingFinished.connect(lambda: signals.append('finished'))
    QTest.keyClick(first if delivery == 'editor' else dialog, key, modifiers)
    assert clicks == ['confirm']
    assert signals == []
    assert dialog.focusWidget() is first


def test_changes_focus_visibility_and_mixed_independent_keys_confirm(scene, ui_qapp):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    for index, editor in enumerate((first, first, second, first, second)):
        if index == 1:
            first.setText('changed')
        if index == 3:
            dialog.hide()
            dialog.show()
            dialog.activateWindow()
        editor.setFocus()
        ui_qapp.processEvents()
        key, modifiers = ((Qt.Key_Return, Qt.NoModifier) if index % 2 == 0
                          else (Qt.Key_Enter, Qt.KeypadModifier))
        QTest.keyClick(editor, key, modifiers)
        assert clicks == ['confirm'] * (index + 1)
    assert first.text() == 'changed' and second.text() == 'second'


def test_same_first_press_setting_reinstall_is_idempotent(scene):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    for _ in range(3):
        assert install_dialog_enter_policy(
            dialog, confirm, confirm_on_first_enter=True,
        ) is policy
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('explicit_default', [False, True])
def test_mode_changes_clear_ready_state_and_same_default_preserves_it(scene, explicit_default):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    native = []
    first.returnPressed.connect(lambda: native.append(True))
    options = {'confirm_on_first_enter': False} if explicit_default else {}
    assert install_dialog_enter_policy(dialog, confirm, **options) is policy
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == [] and native == [True]
    install_dialog_enter_policy(dialog, confirm, confirm_on_first_enter=True)
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm'] and native == [True]
    install_dialog_enter_policy(dialog, confirm, **options)
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm'] and native == [True, True]
    install_dialog_enter_policy(dialog, confirm, **options)
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm', 'confirm'] and native == [True, True]


def test_target_update_clears_old_ready_state_and_uses_new_target(scene):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    install_dialog_enter_policy(dialog, confirm)
    QTest.keyClick(first, Qt.Key_Return)
    install_dialog_enter_policy(dialog, auxiliary)
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['auxiliary']
    install_dialog_enter_policy(dialog, confirm, confirm_on_first_enter=True)
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['auxiliary', 'confirm']


@pytest.mark.parametrize('value', ['5', 'bad'])
def test_validator_rejection_then_acceptable_input(scene, value):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    first.setValidator(QIntValidator(10, 99, first))
    first.setText(value)
    for _ in range(2):
        QTest.keyClick(first, Qt.Key_Return)
    assert clicks == []
    first.setText('42')
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('availability', ['none', 'hidden', 'disabled', 'destroyed'])
def test_unavailable_target_does_not_fall_back(scene, ui_qapp, availability):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    if availability == 'none':
        policy.set_confirm_button(None)
    elif availability == 'hidden':
        confirm.hide()
    elif availability == 'disabled':
        confirm.setEnabled(False)
    else:
        confirm.deleteLater()
        ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    for _ in range(2):
        QTest.keyClick(first, Qt.Key_Return)
    assert clicks == []


@pytest.mark.parametrize('kind', ['spin', 'date', 'combo'])
@pytest.mark.parametrize('delivery', ['inner', 'outer', 'dialog'])
def test_compound_inputs_keep_native_first_then_confirm(scene, ui_qapp, kind, delivery):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    if kind == 'spin':
        editor = QSpinBox()
        editor.setKeyboardTracking(False)
    elif kind == 'date':
        editor = QDateTimeEdit(QDateTime(2026, 10, 6, 12, 30))
    else:
        editor = QComboBox()
        editor.setEditable(True)
        editor.addItems(['first', 'second'])
    dialog.layout().addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    line = editor.findChild(QLineEdit)
    native = []
    line.returnPressed.connect(lambda: native.append('return'))
    if kind != 'combo':
        editor.editingFinished.connect(lambda: native.append('finished'))
    if kind == 'spin':
        line.setText('0042')
    receiver = {'inner': line, 'outer': editor, 'dialog': dialog}[delivery]
    QTest.keyClick(receiver, Qt.Key_Return)
    assert clicks == [] and native
    if kind == 'spin':
        assert editor.value() == 42 and line.text() == '42'
    elif kind == 'date':
        assert editor.dateTime() == QDateTime(2026, 10, 6, 12, 30)
    else:
        assert editor.currentText() == 'first'
    before = list(native)
    QTest.keyClick(line, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ['confirm'] and native == before


@pytest.mark.parametrize('kind', [QComboBox, QTextEdit, QPlainTextEdit])
def test_other_editors_keep_native_behavior(scene, ui_qapp, kind):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    editor = kind()
    if kind is QComboBox:
        editor.addItems(['first', 'second'])
    dialog.layout().addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    for _ in range(2):
        QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    if kind is QComboBox:
        assert editor.currentText() == 'first'
    else:
        assert editor.toPlainText() == '\n\n'


def test_two_windows_keep_mode_and_pending_state_independent(scene, ui_qapp):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    other = QDialog()
    layout = QVBoxLayout(other)
    editor, button = QLineEdit('other'), QPushButton('Other')
    layout.addWidget(editor)
    layout.addWidget(button)
    other_clicks = []
    button.clicked.connect(lambda: other_clicks.append(True))
    install_dialog_enter_policy(other, button)
    try:
        other.show()
        other.activateWindow()
        ui_qapp.processEvents()
        editor.setFocus()
        QTest.keyClick(editor, Qt.Key_Return)
        assert other_clicks == []
        # Send to the other window without stealing editor focus or resetting
        # its pending default-mode confirmation through a native FocusOut.
        QTest.keyClick(first, Qt.Key_Return)
        assert clicks == ['confirm'] and other_clicks == []
        QTest.keyClick(editor, Qt.Key_Return)
        assert other_clicks == [True]
    finally:
        other.close()
        other.deleteLater()
        ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize('delivery', ['editor', 'dialog'])
def test_ime_preedit_and_separate_commit(scene, ui_qapp, delivery):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    ui_qapp.sendEvent(first, QInputMethodEvent('zhong', []))
    receiver = first if delivery == 'editor' else dialog
    QTest.keyClick(receiver, Qt.Key_Return)
    assert clicks == []
    commit = QInputMethodEvent()
    commit.setCommitString('中')
    ui_qapp.sendEvent(first, commit)
    assert first.text().endswith('中')
    QTest.keyClick(receiver, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ['confirm']


@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_ime_commit_during_native_enter_does_not_confirm_same_press(scene, ui_qapp, key, modifiers):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    native = []

    class CandidateCommitEdit(QLineEdit):
        def keyPressEvent(self, event):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                native.append(True)
                commit = QInputMethodEvent()
                commit.setCommitString('中')
                ui_qapp.sendEvent(self, commit)
            super().keyPressEvent(event)

    editor = CandidateCommitEdit()
    dialog.layout().addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    ui_qapp.sendEvent(editor, QInputMethodEvent('zhong', []))
    QTest.keyClick(editor, key, modifiers)
    assert native == [True] and editor.text() == '中' and clicks == []
    QTest.keyClick(editor, key, modifiers)
    assert native == [True] and clicks == ['confirm']


@pytest.mark.parametrize('interaction', ['combo', 'menu', 'modal'])
def test_popup_and_modal_actions_do_not_confirm_parent(scene, ui_qapp, interaction):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    actions = []
    if interaction == 'combo':
        combo = QComboBox()
        combo.addItems(['first', 'second'])
        dialog.layout().addWidget(combo)
        ui_qapp.processEvents()
        combo.setFocus()
        ui_qapp.processEvents()
        combo.showPopup()
        assert QTest.qWaitForWindowExposed(combo.view().window(), 1000)
        ui_qapp.processEvents()
        assert combo.view().isVisible()
        assert QApplication.activePopupWidget() is not None
        QTest.keyClick(first, Qt.Key_Return)
        assert clicks == []
        QTest.keyClick(combo.view(), Qt.Key_Down)
        QTest.keyClick(combo.view(), Qt.Key_Return)
        assert combo.currentText() == 'second' and not combo.view().isVisible()
    else:
        child = QMenu(dialog) if interaction == 'menu' else QDialog(dialog)
        if interaction == 'menu':
            action = child.addAction('Choose')
            action.triggered.connect(lambda: actions.append('chosen'))
            child.popup(dialog.mapToGlobal(dialog.rect().center()))
            child.setActiveAction(action)
        else:
            button = QPushButton('Accept', child)
            QVBoxLayout(child).addWidget(button)
            button.setDefault(True)
            button.clicked.connect(child.accept)
            child.accepted.connect(lambda: actions.append('chosen'))
            child.setModal(True)
            child.show()
        ui_qapp.processEvents()
        QTest.keyClick(first, Qt.Key_Return)
        assert clicks == []
        QTest.keyClick(child, Qt.Key_Return)
        assert actions == ['chosen'] and not child.isVisible()
        child.close()
        child.deleteLater()
    assert clicks == []
    dialog.activateWindow()
    first.setFocus()
    ui_qapp.processEvents()
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_auto_repeat_never_confirms(scene, ui_qapp, key, modifiers):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    for _ in range(3):
        ui_qapp.sendEvent(first, QKeyEvent(QEvent.KeyPress, key, modifiers, '\r', True, 1))
    assert clicks == []
    QTest.keyPress(first, key, modifiers)
    for _ in range(3):
        ui_qapp.sendEvent(first, QKeyEvent(QEvent.KeyPress, key, modifiers, '\r', True, 1))
    assert clicks == ['confirm']
    QTest.keyRelease(first, key, modifiers)
    QTest.keyClick(first, key, modifiers)
    assert clicks == ['confirm', 'confirm']


@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_repeat_cannot_confirm_new_modal_dialog(scene, ui_qapp, key, modifiers):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    child = QMessageBox(QMessageBox.Question, 'Question', 'Proceed?',
                        QMessageBox.Yes | QMessageBox.No, dialog)
    child.setDefaultButton(QMessageBox.Yes)
    child_clicks, observations = [], []
    child.buttonClicked.connect(lambda button: child_clicks.append(button))
    fallback = QTimer(child)
    fallback.setSingleShot(True)
    fallback.timeout.connect(child.reject)

    def exercise_child():
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyPress, key, modifiers, '\r', True, 1))
        observations.append((list(child_clicks), child.isVisible()))
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyRelease, key, modifiers))
        QTest.keyClick(child, key, modifiers)

    def open_child():
        confirm.clicked.disconnect(open_child)
        fallback.start(2000)
        QTimer.singleShot(0, exercise_child)
        child.exec()
        fallback.stop()

    confirm.clicked.connect(open_child)
    QTest.keyClick(first, key, modifiers)
    assert observations == [([], True)]
    assert child_clicks == [child.button(QMessageBox.Yes)]
    assert clicks == ['confirm']
    first.setFocus()
    QTest.keyClick(first, key, modifiers)
    assert clicks == ['confirm', 'confirm']
    child.deleteLater()


@pytest.mark.parametrize('modifier', [Qt.ControlModifier, Qt.AltModifier, Qt.ShiftModifier, Qt.MetaModifier])
def test_modified_enter_does_not_confirm(scene, modifier):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    QTest.keyClick(first, Qt.Key_Return, modifier)
    assert clicks == []
    QTest.keyClick(first, Qt.Key_Return)
    assert clicks == ['confirm']


def test_auxiliary_focus_enter_and_mouse_space_keep_their_targets(scene):
    dialog, first, second, confirm, auxiliary, clicks, policy = scene
    auxiliary.setFocus()
    QTest.keyClick(auxiliary, Qt.Key_Return)
    QTest.mouseClick(auxiliary, Qt.LeftButton)
    QTest.keyClick(auxiliary, Qt.Key_Space)
    assert clicks == ['confirm', 'auxiliary', 'auxiliary']


def test_used_first_press_policy_is_collected_without_callback_errors():
    script = textwrap.dedent('''
        import gc
        import sys
        import traceback
        import weakref
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QVBoxLayout
        from ui.dialog_enter_policy import install_dialog_enter_policy

        app = QApplication([])
        errors, references = [], []
        sys.excepthook = lambda *exc: errors.append(''.join(traceback.format_exception(*exc)))

        def exercise(deferred_delete):
            dialog = QDialog()
            layout = QVBoxLayout(dialog)
            editor, button = QLineEdit('value'), QPushButton('Confirm')
            layout.addWidget(editor)
            layout.addWidget(button)
            policy = install_dialog_enter_policy(dialog, button, confirm_on_first_enter=True)
            references.extend(weakref.ref(obj) for obj in
                              (dialog, layout, editor, button, policy, policy._held_enter))
            dialog.show()
            app.processEvents()
            editor.setFocus()
            # Leave the application guard armed to exercise closing cleanup.
            QTest.keyPress(editor, Qt.Key_Return)
            dialog.close()
            if deferred_delete:
                dialog.deleteLater()
                app.sendPostedEvents(None, QEvent.DeferredDelete)

        for deferred_delete in (False, True):
            for _ in range(10):
                exercise(deferred_delete)
                gc.collect()
                app.processEvents()
                gc.collect()
        assert not errors, errors
        assert all(ref() is None for ref in references), 'First-press objects leaked'
    ''')
    result = subprocess.run(
        [sys.executable, '-B', '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, 'QT_QPA_PLATFORM': 'windows' if sys.platform == 'win32' else 'offscreen'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
