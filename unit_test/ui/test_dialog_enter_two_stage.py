import pytest
from PyQt5 import sip
from PyQt5.QtCore import QDateTime, QEvent, QTimer, Qt
from PyQt5.QtGui import QInputMethodEvent, QIntValidator, QKeyEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QComboBox, QDateTimeEdit, QDialog, QDoubleSpinBox, QLineEdit, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ui.dialog_enter_policy import install_dialog_enter_policy


@pytest.fixture
def scene(ui_qapp):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    editor = QLineEdit('value')
    auxiliary, confirm = QPushButton('Auxiliary'), QPushButton('Confirm')
    for widget in (editor, auxiliary, confirm):
        layout.addWidget(widget)
    native, clicks = [], []
    editor.returnPressed.connect(lambda: native.append(True))
    auxiliary.clicked.connect(lambda: clicks.append('auxiliary'))
    confirm.clicked.connect(lambda: clicks.append('confirm'))
    policy = install_dialog_enter_policy(dialog, confirm)
    dialog.show()
    dialog.activateWindow()
    ui_qapp.processEvents()
    editor.setFocus()
    assert dialog.focusWidget() is editor
    yield dialog, layout, editor, auxiliary, confirm, native, clicks, policy
    dialog.close()
    dialog.deleteLater()
    ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()


def test_first_native_then_confirm(scene):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == []
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == ['confirm']


@pytest.mark.parametrize('keys', [
    [(Qt.Key_Return, Qt.NoModifier)] * 3,
    [(Qt.Key_Enter, Qt.KeypadModifier)] * 3,
    [(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
     (Qt.Key_Return, Qt.NoModifier)],
])
@pytest.mark.parametrize('delivery', ['editor', 'dialog', 'proxy'])
def test_native_signals_then_repeated_business_confirmation(scene, ui_qapp, keys, delivery):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    finished = []
    editor.editingFinished.connect(lambda: finished.append(editor.text()))
    receiver = editor
    if delivery == 'dialog':
        receiver = dialog
    elif delivery == 'proxy':
        receiver = QWidget()
        layout.addWidget(receiver)
        receiver.setFocusProxy(editor)
        ui_qapp.processEvents()
        receiver.setFocus()
        assert dialog.focusWidget() is editor
    QTest.keyClick(receiver, *keys[0])
    assert native == [True] and finished == ['value'] and clicks == []
    for key in keys[1:]:
        QTest.keyClick(receiver, *key)
    assert clicks == ['confirm', 'confirm']
    assert native == [True] and finished == ['value']
    assert dialog.isVisible()  # The business callback may decline to close.


@pytest.mark.parametrize('kind', ['readonly', 'spin', 'double', 'date', 'combo'])
@pytest.mark.parametrize('first_receiver', ['outer', 'inner', 'dialog'])
def test_compound_input_shares_native_then_confirm_state(scene, ui_qapp, kind, first_receiver):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    if kind == 'readonly':
        editor = QLineEdit('C:/recordings')
        editor.setReadOnly(True)
    elif kind == 'spin':
        editor = QSpinBox()
        editor.setKeyboardTracking(False)
    elif kind == 'double':
        editor = QDoubleSpinBox()
        editor.setKeyboardTracking(False)
    elif kind == 'date':
        editor = QDateTimeEdit(QDateTime(2026, 9, 20, 12, 30))
    else:
        editor = QComboBox()
        editor.setEditable(True)
        editor.addItems(['first', 'second'])
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    line = editor if kind == 'readonly' else editor.findChild(QLineEdit)
    if kind in ('spin', 'double'):
        line.setText('0042')
    completed = []
    if hasattr(editor, 'editingFinished'):
        editor.editingFinished.connect(lambda: completed.append(True))
    receiver = {'outer': editor, 'inner': line, 'dialog': dialog}[first_receiver]
    QTest.keyClick(receiver, Qt.Key_Return)
    assert clicks == []
    if kind in ('spin', 'double'):
        assert editor.value() == 42
        assert line.text() == ('42' if kind == 'spin' else '42.00')
    if kind != 'combo':
        assert completed == [True]
    QTest.keyClick(line, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == ['confirm']
    if kind != 'combo':
        assert completed == [True]


@pytest.mark.parametrize('suffix', ['', ' 秒'])
@pytest.mark.parametrize('delivery', ['inner', 'outer', 'dialog', 'proxy'])
def test_spin_preserves_actual_native_receiver_signals(scene, ui_qapp, suffix, delivery):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    spin = QDoubleSpinBox()
    spin.setDecimals(1)
    spin.setSuffix(suffix)
    spin.setValue(4)
    layout.addWidget(spin)
    line = spin.findChild(QLineEdit)
    receiver = {'inner': line, 'outer': spin, 'dialog': dialog}.get(delivery)
    if delivery == 'proxy':
        receiver = QWidget()
        receiver.setFocusProxy(line)
        layout.addWidget(receiver)
    ui_qapp.processEvents()
    line.setFocus()
    assert dialog.focusWidget() is spin
    assert line.focusProxy() is spin
    inner_finished, outer_finished, returns = [], [], []
    line.editingFinished.connect(lambda: inner_finished.append(line.text()))
    spin.editingFinished.connect(lambda: outer_finished.append(spin.value()))
    line.returnPressed.connect(lambda: returns.append(line.text()))
    QTest.keyClick(receiver, Qt.Key_Return)
    # Native QLineEdit delivery emits its own completion before propagating
    # to the spin box. Direct outer delivery only emits outer completion.
    expected_inner = [f'4.0{suffix}'] if delivery in ('inner', 'proxy') else []
    assert inner_finished == expected_inner
    assert outer_finished == [4.0] and clicks == []
    assert len(returns) == (2 if expected_inner else 1)
    QTest.keyClick(line, Qt.Key_Enter, Qt.KeypadModifier)
    assert inner_finished == expected_inner and outer_finished == [4.0]
    assert len(returns) == (2 if expected_inner else 1)
    assert clicks == ['confirm']


@pytest.mark.parametrize('reset', [
    'user_edit', 'program_edit', 'restore_text', 'focus', 'switch_input',
    'hide', 'target', 'destroy',
])
def test_reset_requires_native_enter_again(scene, ui_qapp, reset):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True]
    target_click = 'confirm'
    if reset == 'user_edit':
        QTest.keyClicks(editor, 'x')
    elif reset in ('program_edit', 'restore_text'):
        editor.setText('new')
        if reset == 'restore_text':
            editor.setText('value')
    elif reset == 'focus':
        auxiliary.setFocus()
        editor.setFocus()
    elif reset == 'switch_input':
        other = QLineEdit('other')
        layout.addWidget(other)
        ui_qapp.processEvents()
        other.setFocus()
        QTest.keyClick(other, Qt.Key_Return)
        editor.setFocus()
    elif reset == 'hide':
        dialog.hide()
        dialog.show()
        editor.setFocus()
        ui_qapp.processEvents()
    elif reset == 'target':
        policy.set_confirm_button(auxiliary)
        target_click = 'auxiliary'
    else:
        editor.deleteLater()
        ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(editor)
        editor = QLineEdit('replacement')
        layout.insertWidget(0, editor)
        editor.returnPressed.connect(lambda: native.append(True))
        ui_qapp.processEvents()
        editor.setFocus()
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == [] and native == [True, True]
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == [target_click] and native == [True, True]


def test_same_target_reinstall_keeps_ready_state(scene):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    QTest.keyClick(editor, Qt.Key_Return)
    assert install_dialog_enter_policy(dialog, confirm) is policy
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == ['confirm']


@pytest.mark.parametrize('value', ['5', 'bad'])
def test_validator_rejection_does_not_arm_confirmation(scene, value):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    editor.setValidator(QIntValidator(10, 99, editor))
    editor.setText(value)
    for _ in range(3):
        QTest.keyClick(editor, Qt.Key_Return)
    assert native == [] and clicks == []
    editor.setText('42')
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == []
    QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == ['confirm']


@pytest.mark.parametrize('availability', ['none', 'hidden', 'disabled', 'destroyed'])
def test_unavailable_target_never_falls_back(scene, ui_qapp, availability):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    if availability == 'none':
        policy.set_confirm_button(None)
    elif availability == 'hidden':
        confirm.hide()
    elif availability == 'disabled':
        confirm.setEnabled(False)
    else:
        confirm.deleteLater()
        ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    for _ in range(3):
        QTest.keyClick(editor, Qt.Key_Return)
    assert native == [True] and clicks == []


@pytest.mark.parametrize('interaction', ['focus', 'hide', 'menu', 'modal', 'target'])
def test_native_callback_round_trip_cannot_arm_stale_state(scene, ui_qapp, interaction):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene

    def interrupt_native():
        editor.returnPressed.disconnect(interrupt_native)
        if interaction == 'focus':
            auxiliary.setFocus()
        elif interaction == 'hide':
            dialog.hide()
            dialog.show()
        elif interaction in ('menu', 'modal'):
            child = QMenu(dialog) if interaction == 'menu' else QDialog(dialog)
            if interaction == 'menu':
                child.addAction('Choose')
                child.popup(dialog.mapToGlobal(dialog.rect().center()))
            else:
                child.setModal(True)
                child.show()
            child.close()
            child.deleteLater()
        else:
            policy.set_confirm_button(None)
            policy.set_confirm_button(confirm)
        editor.setFocus()

    editor.returnPressed.connect(interrupt_native)
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    assert dialog.focusWidget() is editor
    count = len(native)
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == [] and len(native) == count + 1
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == ['confirm']


def test_ime_preedit_and_commit_reset_pending_confirmation(scene, ui_qapp):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    QTest.keyClick(editor, Qt.Key_Return)
    ui_qapp.sendEvent(editor, QInputMethodEvent('zhong', []))
    QTest.keyClick(editor, Qt.Key_Return)
    QTest.keyClick(dialog, Qt.Key_Enter, Qt.KeypadModifier)
    assert clicks == []
    commit = QInputMethodEvent()
    commit.setCommitString('中')
    ui_qapp.sendEvent(editor, commit)
    assert editor.text().endswith('中')
    before = len(native)
    QTest.keyClick(editor, Qt.Key_Return)
    assert len(native) == before + 1 and clicks == []
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('interaction', ['combo', 'menu', 'modal'])
def test_temporary_interaction_resets_ready_input(scene, ui_qapp, interaction):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    if interaction == 'combo':
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(['first', 'second'])
        layout.addWidget(combo)
        ui_qapp.processEvents()
        combo.setFocus()
        editor = combo.lineEdit()
    QTest.keyClick(editor, Qt.Key_Return)
    if interaction == 'combo':
        combo.showPopup()
        ui_qapp.processEvents()
        QTest.keyClick(combo.view(), Qt.Key_Down)
        QTest.keyClick(combo.view(), Qt.Key_Return)
        assert combo.currentText() == 'second'
        assert not combo.view().isVisible()
    else:
        child = QMenu(dialog) if interaction == 'menu' else QDialog(dialog)
        if interaction == 'menu':
            action = child.addAction('Choose')
            child.popup(dialog.mapToGlobal(dialog.rect().center()))
            child.setActiveAction(action)
        else:
            child.setModal(True)
            child.show()
        ui_qapp.processEvents()
        QTest.keyClick(editor, Qt.Key_Return)
        assert clicks == []
        if interaction == 'menu':
            QTest.keyClick(child, Qt.Key_Return)
        child.close()
        child.deleteLater()
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('ready', [False, True])
@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_auto_repeat_neither_arms_nor_confirms(scene, ui_qapp, ready, key, modifiers):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    if ready:
        QTest.keyPress(editor, key, modifiers)
    for _ in range(3):
        ui_qapp.sendEvent(editor, QKeyEvent(QEvent.KeyPress, key, modifiers, '\r', True, 1))
    assert clicks == []
    QTest.keyRelease(editor, key, modifiers)
    QTest.keyClick(editor, key, modifiers)
    assert clicks == (['confirm'] if ready else [])


@pytest.mark.parametrize('stage', ['first', 'second'])
@pytest.mark.parametrize('key, modifiers', [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_nested_message_box_requires_new_physical_press(scene, ui_qapp, stage, key, modifiers):
    dialog, layout, editor, auxiliary, confirm, native, clicks, policy = scene
    child = QMessageBox(QMessageBox.Question, 'Question', 'Proceed?',
                        QMessageBox.Yes | QMessageBox.No, dialog)
    child.setDefaultButton(QMessageBox.Yes)
    child_clicks, observations = [], []
    child.buttonClicked.connect(lambda button: child_clicks.append(button))

    def exercise_child():
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyPress, key, modifiers, '\r', True, 1))
        observations.append(list(child_clicks))
        ui_qapp.sendEvent(child, QKeyEvent(QEvent.KeyRelease, key, modifiers))
        QTest.keyClick(child, key, modifiers)
        if child.isVisible():
            child.reject()

    def open_child():
        signal.disconnect(open_child)
        QTimer.singleShot(0, exercise_child)
        child.exec()

    if stage == 'second':
        QTest.keyClick(editor, key, modifiers)
    signal = editor.returnPressed if stage == 'first' else confirm.clicked
    signal.connect(open_child)
    QTest.keyClick(editor, key, modifiers)
    assert observations == [[]]
    assert child_clicks == [child.button(QMessageBox.Yes)]
    assert clicks == ([] if stage == 'first' else ['confirm'])
    editor.setFocus()
    before = len(clicks)
    QTest.keyClick(editor, key, modifiers)
    assert len(clicks) == before
    child.deleteLater()


@pytest.mark.parametrize('kind', ['spin', 'date', 'combo'])
def test_compound_programmatic_value_change_resets(scene, ui_qapp, kind):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    if kind == 'spin':
        editor = QSpinBox()
    elif kind == 'date':
        editor = QDateTimeEdit(QDateTime(2026, 9, 20, 12, 30))
    else:
        editor = QComboBox()
        editor.setEditable(True)
        editor.addItems(['first', 'second'])
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key_Return)
    if kind == 'spin':
        editor.setValue(42)
    elif kind == 'date':
        editor.setDateTime(editor.dateTime().addDays(1))
    else:
        editor.setCurrentIndex(1)
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == ['confirm']


def test_replaced_combo_line_editor_starts_fresh(scene, ui_qapp):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    combo = QComboBox()
    combo.setEditable(True)
    layout.addWidget(combo)
    ui_qapp.processEvents()
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key_Return)
    replacement = QLineEdit('new')
    combo.setLineEdit(replacement)
    ui_qapp.processEvents()
    combo.setFocus()
    QTest.keyClick(replacement, Qt.Key_Return)
    assert clicks == []
    QTest.keyClick(combo, Qt.Key_Return)
    assert clicks == ['confirm']


@pytest.mark.parametrize('persistent', [False, True])
def test_table_editor_commit_lifecycle(scene, ui_qapp, persistent):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    table = QTableWidget(1, 1)
    item = QTableWidgetItem('old')
    table.setItem(0, 0, item)
    layout.addWidget(table)
    ui_qapp.processEvents()
    if persistent:
        table.openPersistentEditor(item)
    else:
        table.editItem(item)
    editor = table.findChild(QLineEdit)
    editor.setFocus()
    editor.setText('new')
    QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    if persistent:
        # While the persistent editor still holds focus, it shares the same
        # second-stage rule. Qt's queued closeEditor can later move focus.
        assert dialog.focusWidget() is editor
        QTest.keyClick(editor, Qt.Key_Return)
        assert clicks == ['confirm']
        ui_qapp.processEvents()
        assert item.text() == 'new'
    else:
        ui_qapp.processEvents()
        assert item.text() == 'new'
        ui_qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(editor)
        QTest.keyClick(table, Qt.Key_Return)
        assert clicks == ['confirm']


@pytest.mark.parametrize('kind', [QComboBox, QTextEdit, QPlainTextEdit])
def test_non_single_line_controls_never_arm_confirmation(scene, ui_qapp, kind):
    dialog, layout, old_editor, auxiliary, confirm, native, clicks, policy = scene
    editor = kind()
    if kind is QComboBox:
        editor.addItems(['first', 'second'])
    layout.addWidget(editor)
    ui_qapp.processEvents()
    editor.setFocus()
    for _ in range(3):
        QTest.keyClick(editor, Qt.Key_Return)
    assert clicks == []
    if kind is not QComboBox:
        assert editor.toPlainText() == '\n\n\n'
