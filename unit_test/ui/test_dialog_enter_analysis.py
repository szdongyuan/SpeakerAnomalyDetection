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


def test_analysis_numeric_second_enter_accepts_committed_config(analysis, dialogs, enter, ui_qapp):
    dialog, manager = analysis
    if isinstance(dialog, FftConfigWindow):
        dialog.channel_selector.combo_box.setCurrentIndex(1)
        spin, text, key, expected = dialog.overlap_spin, "25", "overlap_ratio", 0.25
    else:
        dialog.channel_selector.spin_box.setValue(2)
        if isinstance(dialog, SplConfigWindow):
            dialog.threshold_widget.limit_checkbox.setChecked(True)
            dialog.scalar_upper_check.setChecked(True)
            spin, text, key, expected = dialog.scalar_upper_spin, "73.5", "scalar_upper_value", 73.5
        elif isinstance(dialog, SpecConfigWindow):
            dialog.custom_limit_checkbox.setChecked(True)
            spin, text, key, expected = dialog.top_limit_spinbox, "85", "top_limit", 85
        else:
            spin, text, key, expected = dialog.f_min_spin, "100", "f_min", 100
    dialog.section_scroll_area.ensureWidgetVisible(spin)
    assert spin.isVisible() and spin.isEnabled()
    spin.setKeyboardTracking(False)
    editor = spin.lineEdit()
    editor.setFocus()
    editor.selectAll()
    QTest.keyClicks(editor, text)
    finished, accepted_configs = [], []
    spin.editingFinished.connect(lambda: finished.append(spin.value()))
    dialog.accepted.connect(lambda: accepted_configs.append(dialog.get_default_config()))
    seen = clicks(dialog)
    press(editor, enter, ui_qapp)
    assert spin.value() == float(text)
    assert finished == [float(text)]
    assert seen == []
    assert accepted_configs == []
    assert dialogs[1] == []
    assert dialog.isVisible()
    press(editor, enter, ui_qapp)
    assert seen == [dialog.semantic_ok_btn]
    assert dialog.result() == QDialog.Accepted
    assert not dialog.isVisible()
    assert len(accepted_configs) == 1
    assert accepted_configs[0]["analysis_channel"] == 1
    assert accepted_configs[0][key] == expected
    assert manager.saved == []
    assert dialogs[1] == []


def test_analysis_parameter_editors_first_enter_never_click(analysis, enter, ui_qapp):
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
        # The spinbox and its QLineEdit are both visited; each gets a fresh focus.
        dialog.semantic_cancel_btn.setFocus()
        ui_qapp.processEvents()
        assert dialog.focusWidget() is dialog.semantic_cancel_btn
        dialog.section_scroll_area.ensureWidgetVisible(editor)
        press(editor, enter, ui_qapp)
        assert seen == []
        assert dialog.isVisible()
    assert manager.saved == []


@pytest.mark.parametrize("analysis", [FbaConfigWindow], indirect=True)
def test_analysis_combo_and_multiline_repeated_enter_stays_native(analysis, dialogs, enter, ui_qapp):
    dialog, manager = analysis
    combo = dialog.strategy_combo
    combo.setCurrentText("自定义")
    assert not combo.isEditable()
    editor = dialog.custom_bands_edit
    editor.setPlainText("20, 200")
    seen = clicks(dialog)
    for _ in range(3):
        press(combo, enter, ui_qapp)
        assert combo.currentText() == "自定义"
        assert seen == []
    assert editor.isVisible() and editor.isEnabled()
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key_End, Qt.ControlModifier)
    for count in range(1, 4):
        press(editor, enter, ui_qapp)
        assert editor.toPlainText() == "20, 200" + "\n" * count
        assert seen == []
        assert dialog.isVisible()
    assert dialogs[1] == []
    assert manager.saved == []


@pytest.mark.parametrize("location", ["auxiliary", "second-enter"])
def test_analysis_invalid_limits_still_validate(analysis, dialogs, enter, ui_qapp, location):
    dialog, manager = analysis
    if isinstance(dialog, FftConfigWindow):
        dialog.channel_selector.combo_box.setCurrentIndex(1)
    else:
        dialog.channel_selector.spin_box.setValue(2)
    if isinstance(dialog, SpecConfigWindow):
        dialog.custom_limit_checkbox.setChecked(True)
        dialog.top_limit_spinbox.setValue(20)
        dialog.bottom_limit_spinbox.setValue(30)
        spin = dialog.bottom_limit_spinbox
        warning = "上下限配置数据错误，请检查配置!"
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
        spin = threshold.constant_lower_spin
        warning = "固定上下限配置错误：下限不能大于上限"
    seen = clicks(dialog)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    if location == "second-enter":
        dialog.section_scroll_area.ensureWidgetVisible(spin)
        assert spin.isVisible() and spin.isEnabled()
        finished = []
        spin.editingFinished.connect(lambda: finished.append(spin.value()))
        press(spin.lineEdit(), enter, ui_qapp)
        assert finished == [30]
        assert seen == []
        assert dialogs[1] == []
        assert dialog.isVisible()
        press(spin.lineEdit(), enter, ui_qapp)
    else:
        press(dialog.semantic_cancel_btn, enter, ui_qapp)
    assert seen == [dialog.semantic_ok_btn]
    assert dialog.isVisible()
    assert dialogs[1] == [warning]
    assert accepted == []
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
