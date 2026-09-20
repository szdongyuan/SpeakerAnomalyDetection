"""Opt-in, dialog-owned handling of unmodified Return and keypad Enter."""

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit,
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
    def __init__(self, dialog, confirm_button=None):
        super().__init__(dialog)
        self._dialog = dialog
        self._confirm_button = confirm_button
        self._dispatching_input = False
        self._preedit_widget = None
        self._held_enter = _HeldEnterGuard(dialog)
        self._watch(dialog)

    def set_confirm_button(self, button_or_none):
        self._confirm_button = button_or_none
        self._watch(self._dialog)

    def _watch(self, obj):
        if not isinstance(obj, QWidget) or obj.window() is not self._dialog:
            return
        obj.installEventFilter(self)
        if isinstance(obj, QPushButton):
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

    def eventFilter(self, obj, event):
        if not isinstance(obj, QWidget) or obj.window() is not self._dialog:
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
            self._preedit_widget = obj if event.preeditString() else None
        elif event.type() == QEvent.FocusOut and obj is self._preedit_widget:
            self._preedit_widget = None
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
        if self._is_input(obj):
            self._dispatching_input = True
            try:
                QApplication.sendEvent(obj, event)
            finally:
                self._dispatching_input = False
            return True
        if obj is self._dialog and self._is_input(self._dialog.focusWidget()):
            return True
        if self._preedit_widget is not None:
            return True
        if event.isAutoRepeat():
            return True
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self._dialog:
            return True
        if QApplication.activePopupWidget() is not None:
            return True
        button = self._confirm_button
        if button is not None and not sip.isdeleted(button) and button.isVisible() and button.isEnabled():
            self._held_enter.arm()
            button.click()
        return True


def install_dialog_enter_policy(dialog, confirm_button=None):
    """Install once on *dialog*, or update its explicit confirmation target."""
    policy = getattr(dialog, "_enter_policy", None)
    if policy is None:
        policy = DialogEnterPolicy(dialog, confirm_button)
        dialog._enter_policy = policy
    else:
        policy.set_confirm_button(confirm_button)
    return policy
