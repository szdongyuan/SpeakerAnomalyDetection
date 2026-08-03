"""Shared stylesheet composition for configuration dialogs."""

from consts import ui_style_const


def _without_hash_comments(stylesheet: str) -> str:
    """Remove Python-style comment lines that are invalid in Qt stylesheets."""
    return "\n".join(
        line
        for line in stylesheet.splitlines()
        if not line.lstrip().startswith("#")
    )


CONFIG_DIALOG_BASE_STYLESHEET = (
    ui_style_const.qdialog_style
    + ui_style_const.qlabel_style
    + ui_style_const.qpushbutton_style
    + ui_style_const.qlineedit_style
    + ui_style_const.qcombobox_style
    + ui_style_const.qgroupbox_style
    + ui_style_const.qspinbox_style
    + ui_style_const.qdoublespinbox_style
    + ui_style_const.qcheckbox_style
    + ui_style_const.qradiobutton_style
    + ui_style_const.qtabwidget_style
    + ui_style_const.qtextedit_style
    + ui_style_const.qtableview_style
    + _without_hash_comments(ui_style_const.qtreeview_style)
    + ui_style_const.qlistview_style
)


def build_config_dialog_stylesheet(feature_stylesheet: str = "") -> str:
    """Combine the shared dialog theme with optional feature-specific rules."""
    return CONFIG_DIALOG_BASE_STYLESHEET + feature_stylesheet
