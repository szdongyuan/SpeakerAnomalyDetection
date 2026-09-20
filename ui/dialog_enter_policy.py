"""Opt-in, dialog-owned handling of unmodified Return and keypad Enter."""

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDateTimeEdit, QLineEdit, QPlainTextEdit,
    QPushButton, QTextEdit, QWidget,
)


class _HeldEnterGuard(QObject):
    """Protect a child modal opened by this press until its physical release.

    The application filter exists only during that press. It never dispatches
    confirmation and ignores windows outside this dialog's ownership tree.
    """

    def __init__(self, dialog):
        super().__init__(dialog)
        self._dialog = dialog

    def arm(self):
        QApplication.instance().installEventFilter(self)

    def disarm(self):
        QApplication.instance().removeEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if event.key() not in (Qt.Key_Return, Qt.Key_Enter):
            return False
        if not event.isAutoRepeat():
            self.disarm()
            return False
        if (
            event.modifiers() not in (Qt.NoModifier, Qt.KeypadModifier)
            or not isinstance(obj, QWidget)
        ):
            return False
        owner = obj.window()
        if owner is self._dialog:
            # The policy already blocks button repeats in this window. Native
            # editors must still receive repeats (for example, newlines).
            return False
        while owner is not None:
            if owner is self._dialog:
                return True
            owner = owner.parentWidget()
        return False


class DialogEnterPolicy(QObject):
    """Keep the native first Enter and confirm subsequent unchanged input."""

    def __init__(self, dialog, confirm_button=None):
        super().__init__(dialog)
        self._dialog = dialog
        self._confirm_button = confirm_button
        self._dispatching_input = False
        self._preedit_widget = None
        self._ready_input = None
        self._tracked_input = None
        self._input_epoch = 0
        self._input_connections = []
        self._held_enter = _HeldEnterGuard(dialog)
        self._watch(dialog)

    def set_confirm_button(self, button_or_none):
        if button_or_none is not self._confirm_button:
            self._reset_input()
        self._confirm_button = button_or_none
        self._watch(self._dialog)

    def _watch(self, obj):
        if not isinstance(obj, QWidget):
            return
        obj.installEventFilter(self)
        if obj.window() is self._dialog and isinstance(obj, QPushButton):
            obj.setAutoDefault(False)
            obj.setDefault(obj is self._confirm_button)
        for child in obj.children():
            self._watch(child)

    def _is_input(self, obj):
        while isinstance(obj, QWidget) and obj is not self._dialog:
            if isinstance(obj, (QLineEdit, QAbstractSpinBox, QComboBox)):
                return True
            if isinstance(obj, (QTextEdit, QPlainTextEdit)) and not obj.isReadOnly():
                return True
            obj = obj.parentWidget()
        return False

    def _single_line_input(self, obj):
        if not isinstance(obj, QWidget):
            return None
        while obj.focusProxy() is not None:
            obj = obj.focusProxy()
        while obj is not None and obj is not self._dialog:
            if isinstance(obj, QAbstractSpinBox):
                return obj, obj.findChild(QLineEdit)
            if isinstance(obj, QComboBox):
                return (obj, obj.lineEdit()) if obj.isEditable() else None
            if isinstance(obj, QLineEdit):
                parent = obj.parentWidget()
                if isinstance(parent, (QAbstractSpinBox, QComboBox)):
                    return parent, obj
                return obj, obj
            obj = obj.parentWidget()
        return None

    def _reset_input(self):
        self._ready_input = None
        self._input_epoch += 1

    def _text_changed(self):
        self._ready_input = None
        # Native normalization belongs to the first Enter. Lifecycle changes
        # still advance the epoch while native handling is running.
        if not self._dispatching_input:
            self._input_epoch += 1

    def _input_destroyed(self):
        self._disconnect_input(destroying=self.sender())
        self._reset_input()
        self._tracked_input = None

    def _disconnect_input(self, destroying=None):
        for owner, signal, slot in self._input_connections:
            if owner is not destroying and not sip.isdeleted(owner):
                signal.disconnect(slot)
        self._input_connections.clear()

    def _track_input(self, single_line):
        if single_line == self._tracked_input:
            return
        self._disconnect_input()
        self._reset_input()
        self._tracked_input = single_line
        line = single_line[1]
        self._input_connections = [
            (line, line.textChanged, self._text_changed),
            (line, line.destroyed, self._input_destroyed),
        ]
        owner = single_line[0]
        if isinstance(owner, QDateTimeEdit):
            self._input_connections.append((owner, owner.dateTimeChanged, self._text_changed))
        elif isinstance(owner, QAbstractSpinBox) and hasattr(owner, 'valueChanged'):
            self._input_connections.append((owner, owner.valueChanged, self._text_changed))
        for owner, signal, slot in self._input_connections:
            signal.connect(slot)

    def eventFilter(self, obj, event):
        if not isinstance(obj, QWidget):
            return False
        if obj.window() is not self._dialog:
            # Observe owned temporary windows only for invalidation; their
            # keyboard handling and default buttons remain independent.
            if event.type() == QEvent.Show and (obj.isModal() or obj.windowType() == Qt.Popup):
                self._reset_input()
            return False
        if event.type() == QEvent.ChildAdded:
            # ChildAdded is sent before construction completes: only attach a
            # filter here; inspect widget types/defaults once polished or shown.
            event.child().installEventFilter(self)
        elif event.type() == QEvent.ChildPolished:
            self._watch(event.child())
        elif event.type() == QEvent.Show:
            self._watch(obj)
        elif event.type() == QEvent.InputMethod:
            self._reset_input()
            self._preedit_widget = obj if event.preeditString() else None
        elif event.type() == QEvent.WindowBlocked:
            self._reset_input()
        elif event.type() == QEvent.FocusOut:
            if obj is self._preedit_widget:
                self._preedit_widget = None
            if self._tracked_input is not None and obj in self._tracked_input:
                self._reset_input()
        elif event.type() == QEvent.Hide:
            if obj is self._dialog or (self._tracked_input is not None and obj in self._tracked_input):
                self._reset_input()
        if event.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        # KeypadModifier describes the physical key, not a command shortcut.
        if (
            event.key() not in (Qt.Key_Return, Qt.Key_Enter)
            or event.modifiers() not in (Qt.NoModifier, Qt.KeypadModifier)
        ):
            return False
        if event.type() == QEvent.KeyRelease:
            return False
        modal = QApplication.activeModalWidget()
        if (
            (modal is not None and modal is not self._dialog)
            or QApplication.activePopupWidget() is not None
        ):
            self._reset_input()
            return True
        if self._dispatching_input:
            # Native editors may ignore Enter after emitting editingFinished.
            # Stop the same event at the first non-editor ancestor, even if a
            # callback has changed focus or entered a nested event loop.
            if self._is_input(obj):
                # Arm after sendEvent has traversed the application filters;
                # arming before redispatch would mistake this same physical
                # press for a fresh press and immediately disarm the guard.
                self._held_enter.arm()
                return False
            return True
        receiver = self._dialog.focusWidget() if obj is self._dialog else obj
        # A spin box's inner line edit points its focus proxy back to the
        # outer control. Preserve native delivery to either actual input;
        # only resolve proxy containers that do not handle editing themselves.
        while (
            isinstance(receiver, QWidget)
            and not self._is_input(receiver)
            and receiver.focusProxy() is not None
        ):
            receiver = receiver.focusProxy()
        single_line = self._single_line_input(receiver)
        if single_line is not None:
            self._track_input(single_line)
            if event.isAutoRepeat():
                return True
        if single_line is not None and single_line == self._ready_input:
            self._click_confirm()
            return True
        if self._is_input(receiver):
            epoch = self._input_epoch
            self._dispatching_input = True
            try:
                QApplication.sendEvent(receiver, event)
            finally:
                self._dispatching_input = False
            if (
                single_line is not None
                and epoch == self._input_epoch
                and not sip.isdeleted(single_line[1])
                and self._single_line_input(self._dialog.focusWidget()) == single_line
                and single_line[1].hasAcceptableInput()
                and self._preedit_widget is None
                and not event.isAutoRepeat()
            ):
                self._ready_input = single_line
            return True
        if self._preedit_widget is not None:
            return True
        if event.isAutoRepeat():
            return True
        self._click_confirm()
        return True

    def _click_confirm(self):
        button = self._confirm_button
        if button is not None and not sip.isdeleted(button) and button.isVisible() and button.isEnabled():
            self._held_enter.arm()
            button.click()


def install_dialog_enter_policy(dialog, confirm_button=None):
    """Install once on *dialog*, or update its explicit confirmation target."""
    policy = getattr(dialog, "_enter_policy", None)
    if policy is None:
        policy = DialogEnterPolicy(dialog, confirm_button)
        dialog._enter_policy = policy
    else:
        policy.set_confirm_button(confirm_button)
    return policy
