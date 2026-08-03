"""Base class for dialogs that use the shared configuration theme."""

from PyQt5.QtWidgets import QDialog, QWidget

from ui.config_dialog_theme import build_config_dialog_stylesheet


class ConfigDialogBase(QDialog):
    """Scope the shared configuration theme to one dialog and its children."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.apply_config_dialog_theme()

    def apply_config_dialog_theme(self, feature_stylesheet: str = "") -> None:
        """Apply common rules plus explicit rules owned by the dialog."""
        super().setStyleSheet(build_config_dialog_stylesheet(feature_stylesheet))
