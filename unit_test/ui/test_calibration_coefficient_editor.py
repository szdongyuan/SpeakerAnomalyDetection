"""Real Qt focus and action transactions for the coefficient editor."""

from types import SimpleNamespace

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QObject, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


@pytest.fixture
def host(ui_qapp):
    from ui.calibration_coefficient_editor import CalibrationCoefficientEditor

    window = QWidget()
    layout = QVBoxLayout(window)
    line = QLineEdit()
    other = QLineEdit()
    channel = QComboBox()
    channel.addItems(["Channel 1", "Channel 7"])
    calibrate = QPushButton("Calibrate")
    reset = QPushButton("Reset")
    exit_button = QPushButton("Exit")
    for widget in (line, other, channel, calibrate, reset, exit_button):
        layout.addWidget(widget)
    state = SimpleNamespace(window=window, line=line, other=other, channel=channel,
                            calibrate=calibrate, reset=reset, exit=exit_button,
                            saved=[], errors=[], actions=[], save_ok=True,
                            on_save=None, on_error=None)

    def save(context, factor):
        state.saved.append((context, factor))
        if state.on_save:
            state.on_save()
        return state.save_ok

    def show_error(message):
        state.errors.append(message)
        if state.on_error:
            state.on_error()

    state.editor = CalibrationCoefficientEditor(line, save, show_error, parent=window)
    state.editor.show_value(("device", 1), 1.0)
    window.show()
    window.activateWindow()
    ui_qapp.processEvents()
    line.setFocus()
    ui_qapp.processEvents()
    yield state
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()


def edit(host, text):
    host.line.setFocus()
    host.line.selectAll()
    if text:
        QTest.keyClicks(host.line, text)
    else:
        QTest.keyClick(host.line, Qt.Key_Backspace)


def test_focus_out_commits_once_without_success_popup(host, ui_qapp):
    edit(host, "2.5")
    QTest.keyClick(host.line, Qt.Key_Tab)
    ui_qapp.processEvents()
    assert not host.line.hasFocus()
    assert host.saved == [(("device", 1), 2.5)]
    assert host.line.text() == "2.5"
    assert host.errors == []
    host.other.clearFocus()
    ui_qapp.processEvents()
    assert len(host.saved) == 1


def test_real_mouse_focus_out_precedes_target_widget_press_filter(host):
    events = []

    class Trace(QObject):
        def eventFilter(self, watched, event):
            if watched is host.line and event.type() == QEvent.FocusOut:
                events.append(("focus-out", QApplication.focusWidget(), event.reason()))
            elif watched is host.calibrate and event.type() == QEvent.MouseButtonPress:
                events.append(("press", QApplication.focusWidget(), None))
            return False

    trace = Trace(host.window)
    host.line.installEventFilter(trace)
    host.calibrate.installEventFilter(trace)
    edit(host, "2.5")
    QTest.mouseClick(host.calibrate, Qt.LeftButton)
    assert events == [("focus-out", host.calibrate, Qt.MouseFocusReason),
                      ("press", host.calibrate, None)]


@pytest.mark.parametrize("name", ["calibrate", "reset"])
@pytest.mark.parametrize("failure", ["input", "backend"])
def test_failed_click_cancels_same_action_but_next_click_works(host, name, failure):
    button = getattr(host, name)
    host.editor.register_action(button, name)
    button.clicked.connect(lambda: host.actions.append(name)
                           if host.editor.before_action(name) else None)
    host.save_ok = failure != "backend"
    edit(host, "invalid" if failure == "input" else "2.5")
    QTest.mouseClick(button, Qt.LeftButton)
    assert host.actions == []
    assert host.line.text() == "1.0"
    assert len(host.saved) == (0 if failure == "input" else 1)
    assert len(host.errors) == (1 if failure == "input" else 0)
    QTest.mouseClick(button, Qt.LeftButton)
    assert host.actions == [name]


def test_programmatic_action_preflight_and_failed_exit(host):
    edit(host, "invalid")
    assert host.editor.before_action("reset") is False
    assert host.errors
    assert host.editor.before_action("reset") is True
    edit(host, "invalid")
    assert host.editor.before_action("exit") is True
    assert len(host.errors) == 2


def wire_channel(host):
    host.editor.register_action(host.channel, "channel")

    def select(index):
        if host.editor.before_action("channel"):
            host.actions.append(index)
            host.editor.show_value(("device", [1, 7][index]), [1.0, 7.0][index])

    host.channel.currentIndexChanged.connect(select)


@pytest.mark.parametrize("save_ok", [True, False])
def test_combo_click_saves_old_context_before_channel_signal(host, ui_qapp, save_ok):
    wire_channel(host)
    host.save_ok = save_ok
    edit(host, "2.5")
    QTest.mouseClick(host.channel, Qt.LeftButton)
    ui_qapp.processEvents()
    assert host.saved == [(("device", 1), 2.5)]
    if not save_ok:
        assert not host.channel.view().isVisible()
        assert host.channel.currentIndex() == 0
        assert host.actions == []
        QTest.mouseClick(host.channel, Qt.LeftButton)
    assert host.channel.view().isVisible()
    QTest.keyClick(host.channel.view(), Qt.Key_Down)
    QTest.keyClick(host.channel.view(), Qt.Key_Enter)
    ui_qapp.processEvents()
    assert host.channel.currentIndex() == 1
    assert host.actions == [1]
    assert host.line.text() == "7.0"
    assert host.saved == [(("device", 1), 2.5)]


@pytest.mark.parametrize("failure", ["input", "backend"])
def test_modal_failure_restores_before_nested_focus_and_cancels_click(host, failure):
    host.editor.register_action(host.reset, "reset")
    host.reset.clicked.connect(lambda: host.actions.append("reset"))
    reentrant_results = []

    def nested_dialog():
        assert host.line.text() == "1.0"
        reentrant_results.append(host.editor.commit_pending())
        dialog = QDialog(host.window)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLineEdit())
        QTimer.singleShot(0, dialog.accept)
        dialog.exec_()
        dialog.deleteLater()

    if failure == "input":
        host.on_error = nested_dialog
    else:
        host.save_ok = False
        host.on_save = nested_dialog
    edit(host, "bad" if failure == "input" else "2.5")
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.actions == []
    assert reentrant_results == [False]
    assert len(host.errors) == (1 if failure == "input" else 0)
    assert len(host.saved) == (0 if failure == "input" else 1)
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.actions == ["reset"]


def test_failed_plain_tab_leaves_no_action_suppression(host):
    host.editor.register_action(host.reset, "reset")
    edit(host, "bad")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.errors
    assert host.editor.before_action("reset") is True
    host.reset.clicked.connect(lambda: host.actions.append("reset"))
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.actions == ["reset"]


def test_exit_click_is_allowed_after_failed_save(host):
    host.editor.register_action(host.exit, "exit")
    host.exit.clicked.connect(lambda: host.window.close()
                              if host.editor.before_action("exit") else None)
    edit(host, "bad")
    QTest.mouseClick(host.exit, Qt.LeftButton)
    assert len(host.errors) == 1
    assert not host.window.isVisible()


@pytest.mark.parametrize("text", ["  2.5  ", "3.141592653589793", "2.5e-23", "5e-324"])
def test_positive_numbers_round_trip_without_precision_loss(host, text):
    edit(host, text)
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == [(("device", 1), float(text))]
    assert float(host.line.text()) == float(text)
    assert host.line.text() == repr(float(text))


@pytest.mark.parametrize("text", ["", "bad", "0", "-1", "nan", "inf", "1e9999", "1e-9999"])
@pytest.mark.parametrize("original", [None, 1.2345678901234567])
def test_invalid_edited_value_restores_confirmed_value(host, text, original):
    host.editor.show_value(("device", 1), original)
    # Empty initial input needs a real edit before deletion creates a draft.
    edit(host, "2")
    edit(host, text)
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == []
    assert len(host.errors) == 1
    assert host.line.text() == ("" if original is None else repr(original))


@pytest.mark.parametrize("original", [None, 1.2345678901234567, 5e-324])
def test_display_and_unedited_focus_do_not_save(host, original):
    host.editor.show_value(("device", 7), original)
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == []
    assert host.errors == []
    assert host.line.text() == ("" if original is None else repr(original))


def test_same_numeric_value_normalizes_without_saving(host):
    edit(host, " 1e0 ")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == []
    assert host.line.text() == "1.0"


def test_programmatic_text_change_does_not_create_draft(host):
    host.line.setText("9.0")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == []


@pytest.mark.parametrize("invalidate", ["disable", "discard", "new-context"])
def test_invalidation_before_focus_loss_drops_old_draft(host, invalidate):
    edit(host, "2.5")
    if invalidate == "disable":
        host.editor.set_editable(False)
        host.line.setEnabled(False)
    elif invalidate == "discard":
        host.editor.discard_pending()
    else:
        host.editor.show_value(("other-device", 7), 7.0)
    QTest.mouseClick(host.other, Qt.LeftButton)
    assert host.saved == []
    assert host.errors == []
    assert host.line.text() == ("7.0" if invalidate == "new-context" else "1.0")


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_return_does_not_save_or_activate_action(host, key):
    for action in ("calibrate", "reset", "exit"):
        button = getattr(host, action)
        host.editor.register_action(button, action)
        button.clicked.connect(lambda: host.actions.append(True))
    edit(host, "2.5")
    QTest.keyClick(host.line, key)
    assert host.saved == []
    assert host.actions == []
    assert host.line.text() == "2.5"
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.saved == [(("device", 1), 2.5)]


@pytest.mark.parametrize("ending", ["explicit", "parent-close", "delete-editor"])
def test_editor_lifecycle_detaches_filters_without_saving(host, ui_qapp, ending):
    host.editor.register_action(host.reset, "reset")
    host.reset.clicked.connect(lambda: host.actions.append("reset"))
    edit(host, "2.5")
    if ending == "explicit":
        host.editor.close()
        host.editor.close()  # teardown can be reached by multiple close paths
    elif ending == "parent-close":
        host.editor.discard_pending()  # forced teardown invalidates first
        host.window.close()
        host.window.show()
        host.window.activateWindow()
        ui_qapp.processEvents()
    else:
        host.editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(host.editor)
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.saved == []
    assert host.actions == ["reset"]


def test_normal_owner_close_commits_before_detaching(host):
    edit(host, "2.5")
    host.window.close()
    assert host.saved == [(("device", 1), 2.5)]
    assert host.errors == []


def test_destroyed_action_does_not_break_editor_close(host):
    host.editor.register_action(host.reset, "reset")
    host.reset.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    host.editor.close()


def test_no_focus_action_commits_and_cancels_before_press_signal(host):
    host.reset.setFocusPolicy(Qt.NoFocus)
    host.editor.register_action(host.reset, "reset")
    host.reset.pressed.connect(lambda: host.actions.append("pressed"))
    host.reset.clicked.connect(lambda: host.actions.append("clicked"))
    edit(host, "bad")
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.actions == []
    assert len(host.errors) == 1
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.actions == ["pressed", "clicked"]


def test_failed_drag_release_elsewhere_does_not_block_later_program_action(host, ui_qapp):
    host.editor.register_action(host.reset, "reset")
    edit(host, "bad")
    QTest.mousePress(host.reset, Qt.LeftButton)
    assert QApplication.mouseButtons() == Qt.LeftButton
    assert host.editor.before_action("reset") is False
    QTest.mouseRelease(host.other, Qt.LeftButton)
    ui_qapp.processEvents()
    assert QApplication.mouseButtons() == Qt.NoButton
    assert host.editor.before_action("reset") is True


def test_reopened_owner_has_no_active_editor_callbacks(host, ui_qapp):
    host.editor.register_action(host.reset, "reset")
    host.window.close()
    host.window.show()
    host.window.activateWindow()
    ui_qapp.processEvents()
    edit(host, "2.5")
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.saved == []


def test_unregistered_foreign_action_is_not_cancelled(host):
    host.reset.clicked.connect(lambda: host.actions.append("reset"))
    edit(host, "bad")
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert host.errors
    assert host.actions == ["reset"]


def test_backend_failure_keeps_baseline_until_later_success(host):
    host.save_ok = False
    edit(host, "2.5")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.line.text() == "1.0"
    host.save_ok = True
    edit(host, "3.5")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.line.text() == "3.5"
    edit(host, "bad")
    QTest.keyClick(host.line, Qt.Key_Tab)
    assert host.line.text() == "3.5"
    assert host.saved == [(("device", 1), 2.5), (("device", 1), 3.5)]


@pytest.mark.parametrize("takes_focus", [True, False])
def test_successful_action_saves_before_pressed_signal(host, takes_focus):
    if not takes_focus:
        host.reset.setFocusPolicy(Qt.NoFocus)
    host.editor.register_action(host.reset, "reset")
    seen = []
    host.reset.pressed.connect(lambda: seen.append(list(host.saved)))
    host.reset.clicked.connect(lambda: host.actions.append("reset")
                               if host.editor.before_action("reset") else None)
    edit(host, "2.5")
    QTest.mouseClick(host.reset, Qt.LeftButton)
    assert seen == [[(("device", 1), 2.5)]]
    assert host.actions == ["reset"]
    assert host.saved == [(("device", 1), 2.5)]


def test_programmatic_mouse_reason_focus_is_not_a_pending_mouse_gesture(host):
    host.editor.register_action(host.reset, "reset")
    edit(host, "bad")
    host.reset.setFocus(Qt.MouseFocusReason)
    assert len(host.errors) == 1
    assert host.editor.before_action("reset") is True


@pytest.mark.parametrize("failure", ["input", "backend"])
def test_normal_owner_close_finishes_even_if_commit_fails(host, failure):
    host.save_ok = failure != "backend"
    edit(host, "bad" if failure == "input" else "2.5")
    host.window.close()
    assert not host.window.isVisible()
    assert host.line.text() == "1.0"
    assert len(host.saved) == (1 if failure == "backend" else 0)
    assert len(host.errors) == (1 if failure == "input" else 0)
