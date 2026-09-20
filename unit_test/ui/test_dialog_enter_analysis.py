"""Real keyboard integration for supported analysis dialogs and their editors."""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from ui.custom_ui_widget.audio_clip_extraction_dialog import AudioClipExtractionDialog
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import MessageBox
from ui.output_load_config_dialog import OutputLoadConfigDialog
from ui.ui_analysis_config.common_widgets import AnalysisConfigDialogBase
from ui.ui_analysis_config.curve_color_config_widget import PresetColorDialog
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.ui_analysis_config.threshold_config_widget import _ManualLimitEditorDialog


class ConfigManager:
    def __init__(self, name):
        self.config = {name: {"limit_checked": False}}
        self.saved = []

    def load_config(self):
        return self.config

    def save_default_config(self, name, config):
        self.saved.append((name, config))
        return True


@pytest.fixture(params=[(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier)], ids=["return", "keypad"])
def enter(request):
    return request.param


@pytest.fixture
def dialogs(ui_qapp, monkeypatch):
    opened = []
    warnings = []
    monkeypatch.setattr(MessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(PopupUtils, "save_popup", lambda *args, **kwargs: None)

    def native(*args, **kwargs):
        raise AssertionError("Enter opened a native file selector")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", native)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", native)

    def show(dialog):
        if dialog not in opened:
            opened.append(dialog)
        dialog.show()
        dialog.activateWindow()
        assert QTest.qWaitForWindowActive(dialog)
        ui_qapp.processEvents()
        return dialog

    yield show, warnings
    for dialog in reversed(opened):
        dialog.close()
        dialog.deleteLater()
    ui_qapp.processEvents()


def press(widget, enter, app):
    widget.setFocus()
    app.processEvents()
    # QTest dispatches to this window even if Windows foreground focus changes.
    focused = widget.window().focusWidget()
    assert focused is widget or (widget.focusProxy() is not None and focused is widget.focusProxy())
    QTest.keyClick(widget, *enter)
    app.processEvents()


def clicks(dialog):
    result = []
    for button in dialog.findChildren(QPushButton):
        button.clicked.connect(lambda checked=False, button=button: result.append(button))
    return result


@pytest.fixture(params=[SplConfigWindow, SpecConfigWindow, FbaConfigWindow, FftConfigWindow], ids=["SPL", "Spec", "FBA", "FFT"])
def analysis(request, dialogs):
    name = {SplConfigWindow: "SPL", SpecConfigWindow: "Spec", FbaConfigWindow: "FBA", FftConfigWindow: "FFT"}[request.param]
    manager = ConfigManager(name)
    dialog = request.param(manager, name, available_channels=[0, 1])
    dialogs[0](dialog)
    return dialog, manager


@pytest.mark.parametrize("location", ["default", "restore", "cancel", "navigation", "background"])
def test_analysis_auxiliary_enter_uses_existing_confirmation(analysis, dialogs, enter, ui_qapp, location):
    dialog, manager = analysis
    if isinstance(dialog, FftConfigWindow):
        dialog.channel_selector.combo_box.setCurrentIndex(1)
    else:
        dialog.channel_selector.spin_box.setValue(2)
    seen = clicks(dialog)
    accepted_configs = []
    dialog.accepted.connect(lambda: accepted_configs.append(dialog.get_default_config()))
    target = {
        "default": dialog.semantic_default_btn,
        "restore": dialog.semantic_restore_btn,
        "cancel": dialog.semantic_cancel_btn,
        "navigation": list(dialog._semantic_nav_buttons.values())[-1],
        "background": dialog,
    }[location]
    if location == "background":
        dialog.hide()
        dialogs[0](dialog)
    press(target, enter, ui_qapp)
    assert seen == [dialog.semantic_ok_btn]
    assert dialog.result() == QDialog.Accepted
    assert accepted_configs[0]["analysis_channel"] == 1
    assert manager.saved == []
    assert dialogs[1] == []
    assert dialog.semantic_ok_btn.isDefault()


def test_analysis_parameter_editors_never_click(analysis, enter, ui_qapp):
    dialog, manager = analysis
    if isinstance(dialog, SpecConfigWindow):
        dialog.custom_limit_checkbox.setChecked(True)
    elif isinstance(dialog, SplConfigWindow):
        dialog.analysis_time_range_widget.enabled_checkbox.setChecked(True)
        dialog.threshold_widget.limit_checkbox.setChecked(True)
        dialog.scalar_lower_check.setChecked(True)
    else:
        threshold = dialog.threshold_widget
        threshold.limit_checkbox.setChecked(True)
        threshold.manual_mode_radio.setChecked(True)
        threshold.manual_input_combo.setCurrentIndex(threshold.manual_input_combo.findData("constant"))
        threshold.constant_lower_check.setChecked(True)
    seen = clicks(dialog)
    editors = [widget for widget in dialog.findChildren((QAbstractSpinBox, QComboBox, QLineEdit))
               if widget.isVisible() and widget.isEnabled()]
    channel = (dialog.channel_selector.combo_box if isinstance(dialog, FftConfigWindow)
               else dialog.channel_selector.spin_box)
    assert channel in editors
    for editor in editors:
        dialog.section_scroll_area.ensureWidgetVisible(editor)
        press(editor, enter, ui_qapp)
        assert seen == []
        assert dialog.isVisible()
    assert manager.saved == []


def test_analysis_invalid_limits_still_validate(analysis, dialogs, enter, ui_qapp):
    dialog, manager = analysis
    if isinstance(dialog, SpecConfigWindow):
        dialog.custom_limit_checkbox.setChecked(True)
        dialog.top_limit_spinbox.setValue(20)
        dialog.bottom_limit_spinbox.setValue(30)
    else:
        if isinstance(dialog, SplConfigWindow):
            dialog.limit_metric_combo.setCurrentIndex(dialog.limit_metric_combo.findData("curve_y"))
        threshold = dialog.threshold_widget
        threshold.limit_checkbox.setChecked(True)
        threshold.manual_mode_radio.setChecked(True)
        threshold.manual_input_combo.setCurrentIndex(threshold.manual_input_combo.findData("constant"))
        threshold.constant_upper_check.setChecked(True)
        threshold.constant_lower_check.setChecked(True)
        threshold.constant_upper_spin.setValue(20)
        threshold.constant_lower_spin.setValue(30)
    seen = clicks(dialog)
    press(dialog.semantic_cancel_btn, enter, ui_qapp)
    assert seen == [dialog.semantic_ok_btn]
    assert dialog.isVisible()
    assert dialogs[1]
    assert manager.saved == []


def test_standard_footer_uses_confirmation_and_protects_editor(dialogs, enter, ui_qapp):
    dialog = AnalysisConfigDialogBase()
    calls = []
    layout = QVBoxLayout(dialog)
    editor = QLineEdit()
    layout.addWidget(editor)
    footer = dialog.create_standard_button_layout(lambda: calls.append("default"), lambda: calls.append("ok"))
    layout.addLayout(footer)
    dialogs[0](dialog)
    press(editor, enter, ui_qapp)
    assert calls == []
    press(footer.itemAt(0).widget(), enter, ui_qapp)
    assert calls == ["ok"]


@pytest.mark.parametrize("valid", [True, False])
def test_manual_limits_confirm_without_export(dialogs, enter, ui_qapp, valid):
    config = {"manual_upper_enabled": True, "manual_lower_enabled": False,
              "manual_upper_segments": [{"start_x": 0, "start_y": 10, "end_x": 1, "end_y": 20}] if valid else [],
              "manual_lower_segments": []}
    dialog = dialogs[0](_ManualLimitEditorDialog(None, config, "FFT"))
    seen = clicks(dialog)
    press(dialog.export_button, enter, ui_qapp)
    assert seen == [dialog.confirm_button]
    if valid:
        assert dialog.result() == QDialog.Accepted
        assert dialog.manual_config()["manual_upper_segments"] == config["manual_upper_segments"]
    else:
        assert dialog.isVisible()
        assert dialogs[1]
        assert dialog.manual_config() == {}


def test_manual_table_editor_commits_without_confirming(dialogs, enter, ui_qapp):
    dialog = dialogs[0](_ManualLimitEditorDialog(None, {"manual_upper_enabled": True}, "SPL"))
    table = dialog.editor.manual_upper_table
    table.setCurrentCell(0, 0)
    table.editItem(table.item(0, 0))
    ui_qapp.processEvents()
    editor = table.findChild(QLineEdit)
    assert editor is not None
    editor.setText("0.25")
    seen = clicks(dialog)
    press(editor, enter, ui_qapp)
    assert table.item(0, 0).text() == "0.25"
    assert seen == []
    assert dialog.isVisible()


def test_palette_enter_confirms_without_selecting_focused_color(dialogs, enter, ui_qapp):
    dialog = dialogs[0](PresetColorDialog("#ff0000"))
    other_color, button = next((color, button) for color, button in dialog._color_buttons.items() if color != dialog.selected_color)
    original = dialog.selected_color
    seen = clicks(dialog)
    press(button, enter, ui_qapp)
    assert seen == [dialog.confirm_button]
    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_color == original
    assert dialog.selected_color != other_color


@pytest.mark.parametrize("valid", [True, False])
def test_segmentation_cancel_focus_confirms_only_valid_settings(dialogs, enter, ui_qapp, valid):
    dialog = dialogs[0](OutputLoadConfigDialog("test", {"mode": "time", "interval_seconds": 2, "analysis_seconds": 1}, 10 if valid else None))
    seen = clicks(dialog)
    press(dialog.interval_input.lineEdit(), enter, ui_qapp)
    assert seen == []
    press(dialog.buttons.button(QDialogButtonBox.Cancel), enter, ui_qapp)
    if valid:
        assert seen == [dialog.buttons.button(QDialogButtonBox.Ok)]
        assert dialog.result() == QDialog.Accepted
        assert dialog.settings()["interval_seconds"] == 2
    else:
        assert seen == []
        assert dialog.isVisible()
        assert dialog.error_label.text()


@pytest.mark.parametrize("valid", [True, False])
def test_clip_enter_confirms_without_loading_audio(dialogs, enter, ui_qapp, valid):
    dialog = dialogs[0](AudioClipExtractionDialog(sample_rate=10))
    dialog.audio_data = np.arange(20, dtype=float)
    if valid:
        dialog.selected_region_time = (0.2, 0.5)
    seen = clicks(dialog)
    press(dialog.file_path_edit, enter, ui_qapp)
    assert seen == []
    load = dialog.open_file_layout.itemAt(0).widget()
    press(load, enter, ui_qapp)
    assert len(seen) == 1
    assert seen[0] is not load
    if valid:
        assert dialog.result() == QDialog.Accepted
        np.testing.assert_array_equal(dialog.return_value[0], [2, 3, 4])
        assert dialog.return_value[2] == 3
    else:
        assert dialog.isVisible()
        assert dialogs[1] == ["请选择音频片段"]
