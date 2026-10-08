"""A panel-owned coefficient draft; persistence remains in the panel adapter."""

import math

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QApplication


def parse_coefficient(text):
    """Parse the existing floating-point Pa/V contract without rounding."""
    try:
        value = float(text.strip())
    except (ValueError, OverflowError) as exc:
        raise ValueError("请输入大于零的有效校准系数") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("校准系数必须是大于零的有限数值")
    return value


class CalibrationCoefficientEditor(QObject):
    """Own one draft bound to an immutable device/channel context supplied by UI.

    ``save(context, factor)`` must return a bool and handle backend errors.
    ``show_error(message)`` is used only for invalid input.
    """

    def __init__(self, line_edit, save, show_error, *, parent=None):
        super().__init__(parent if parent is not None else line_edit)
        self._line = line_edit
        self._save = save
        self._show_error = show_error
        self._context = None
        self._factor = None
        self._dirty = False
        self._committing = False
        self._editable = True
        self._closed = False
        self._actions = {}
        self._blocked_target = None
        self._awaiting_press = False
        line_edit.textEdited.connect(self._text_edited)
        line_edit.installEventFilter(self)
        self.parent().installEventFilter(self)

    def _text_edited(self, _text):
        if self._editable and not self._committing:
            self._dirty = True

    def _restore_text(self):
        self._line.setText("" if self._factor is None else repr(float(self._factor)))

    def show_value(self, context, factor):
        """Replace the context and confirmed value without creating a draft."""
        self._context = context
        self._factor = factor
        self.discard_pending()

    def discard_pending(self):
        self._dirty = False
        self._restore_text()

    def set_editable(self, enabled):
        if self._closed:
            return
        self._editable = bool(enabled)
        if not enabled:
            self.discard_pending()
        self._line.setReadOnly(not enabled)

    def register_action(self, widget, action):
        """Guard this panel's button/combo mouse transaction before its signals."""
        if self._closed:
            return
        self._actions[widget] = action
        widget.installEventFilter(self)

    def close(self):
        """Discard and detach on forced teardown; normal exits preflight first.

        Deleting this QObject also makes Qt remove its filters/connections.
        No QApplication filter or global editor state is installed.
        """
        if self._closed:
            return
        self._closed = True
        self._editable = False
        self._dirty = False
        if not sip.isdeleted(self._line):
            self._restore_text()
            self._line.setReadOnly(True)
            self._line.textEdited.disconnect(self._text_edited)
            self._line.removeEventFilter(self)
        for widget in self._actions:
            if not sip.isdeleted(widget):
                widget.removeEventFilter(self)
        self._actions.clear()
        self._blocked_target = None
        self._awaiting_press = False
        self.parent().removeEventFilter(self)

    def before_action(self, action):
        """Use at every action entry, including programmatic and close paths.

        ``exit`` always permits closing after reporting/restoring a failed draft.
        All other action names require a successful commit.
        """
        # A swallowed press does not establish Qt's normal mouse grab. Its
        # release can arrive at an unrelated widget after dragging away. Once
        # that gesture has ended it must not block a later programmatic action.
        if not self._awaiting_press and QApplication.mouseButtons() == Qt.NoButton:
            self._blocked_target = None
        if self._blocked_target is not None and action != "exit":
            return False
        committed = self.commit_pending()
        return committed or action == "exit"

    def commit_pending(self):
        """Save once, restoring confirmed text before callbacks can open dialogs."""
        if self._committing:
            return False
        if not self._dirty or not self._editable:
            return True
        text = self._line.text()
        self._dirty = False
        self._restore_text()
        self._committing = True
        try:
            try:
                factor = parse_coefficient(text)
            except ValueError as exc:
                self._show_error(str(exc))
                return False
            if factor == self._factor:
                return True
            if not self._save(self._context, factor):
                return False
            self._factor = factor
            self._restore_text()
            return True
        finally:
            self._committing = False

    def eventFilter(self, watched, event):
        if watched is self.parent() and event.type() == QEvent.Close:
            self.before_action("exit")
            self.close()
        elif watched is self._line and event.type() == QEvent.FocusOut:
            target = QApplication.focusWidget()
            # QApplication moves focus before dispatching the press to the
            # target widget's filters. Capture that target before a save/error
            # callback can enter a nested dialog event loop and move focus again.
            mouse_action = (event.reason() == Qt.MouseFocusReason
                            and QApplication.mouseButtons() != Qt.NoButton
                            and target in self._actions
                            and self._actions[target] != "exit")
            if mouse_action:
                self._blocked_target = target
                self._awaiting_press = True
            if self.commit_pending() and mouse_action:
                self._blocked_target = None
                self._awaiting_press = False
        elif watched in self._actions:
            if event.type() == QEvent.MouseButtonPress:
                if self._blocked_target is watched and self._awaiting_press:
                    self._awaiting_press = False
                    return True
                # A new independent press ends any prior interrupted gesture.
                self._blocked_target = None
                self._awaiting_press = False
                if not self.before_action(self._actions[watched]):
                    self._blocked_target = watched
                    return True
            elif event.type() == QEvent.MouseButtonRelease and self._blocked_target is watched:
                self._blocked_target = None
                self._awaiting_press = False
                return True
        return super().eventFilter(watched, event)
