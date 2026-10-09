"""Localized confirmation prompts with explicit actions and Qt return values."""

from PyQt5.QtWidgets import QDialogButtonBox, QMessageBox, QProxyStyle, QStyle


class _ConfirmationButtonStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_DialogButtonLayout:
            return QDialogButtonBox.GnomeLayout
        return super().styleHint(hint, option, widget, returnData)


def set_confirmation_button_order(dialog):
    # QMessageBox rebuilds its button layout from its own style hint on show.
    # Styling only its button box is insufficient when a parent has a stylesheet.
    style = _ConfirmationButtonStyle()
    style.setParent(dialog)
    dialog._confirmation_button_style = style
    dialog.setStyle(style)
    dialog.findChild(QDialogButtonBox).setStyle(style)


class ConfirmationMessageBox(QMessageBox):
    def __init__(self, parent, title, text, buttons, default_button):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(text)
        self.setIcon(QMessageBox.Question)
        self.setStandardButtons(buttons)
        set_confirmation_button_order(self)
        for standard, label in (
            (QMessageBox.Yes, "确认"),
            (QMessageBox.No, "取消"),
            (QMessageBox.Save, "保存"),
            (QMessageBox.Discard, "不保存"),
            (QMessageBox.Cancel, "取消"),
        ):
            button = self.button(standard)
            if button is not None:
                button.setText(label)
        self.setDefaultButton(default_button)
        self.setEscapeButton(
            QMessageBox.Cancel if buttons & QMessageBox.Cancel else QMessageBox.No
        )

    @classmethod
    def question(cls, parent, title, text, buttons, default_button):
        dialog = cls(parent, title, text, buttons, default_button)
        return dialog.exec_()
