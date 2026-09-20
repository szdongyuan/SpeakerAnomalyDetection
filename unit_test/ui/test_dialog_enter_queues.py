"""Real-dialog Enter routing with all persistence confined to tmp_path."""

import json
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QLineEdit,
    QMessageBox, QPushButton,
)

from base.sequence_queue_references import SequenceQueueReferenceScanner
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_project_config_dialog import (
    ProductTestProjectConfigDialog, _CopyConditionsDialog,
)
from ui.product_test_program_config_dialog import ProductTestProgramConfigDialog
from unit_test.test_product_test_project_config_dialog import (
    make_manager as make_project_manager, prepare_project,
)
from unit_test.test_product_test_program_config_dialog import (
    make_manager as make_program_manager, prepare_program,
)


@pytest.fixture(params=[(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier)],
                ids=["return", "keypad-enter"])
def enter(request):
    return request.param


@pytest.fixture(autouse=True)
def isolate_prompts(monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a: messages.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: messages.append(a[2]))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Cancel)
    monkeypatch.setattr(QFileDialog, "exec_", lambda self: QDialog.Rejected)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    return messages


def show(dialog, app):
    dialog.show()
    dialog.activateWindow()
    app.processEvents()


def press(dialog, widget, enter, app):
    widget.setFocus()
    app.processEvents()
    assert dialog.focusWidget() is (widget.focusProxy() or widget)
    QTest.keyClick(widget, *enter)
    app.processEvents()


def clicks_for(dialog):
    clicks = []
    for button in dialog.findChildren(QPushButton):
        button.clicked.connect(lambda checked=False, button=button: clicks.append(button))
    return clicks


def dispose(dialog, app):
    dialog._allow_close = True
    dialog._dirty = False
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


@pytest.fixture
def queue_dialog(ui_qapp, tmp_path, monkeypatch):
    target = tmp_path / "queue.json"
    registry = tmp_path / "queues.json"
    payload = [{"seq1": {"acq": {"name": "录制音频", "mode": "RECORD_ONLY",
                "detail": {"sample_rate": 44100, "total_time": 4.0}},
                "analysis_list": {"display_sequence": [], "default_ai": None,
                                  "auto_analysis": True}}}]
    target.write_text(json.dumps(payload), encoding="utf-8")
    registry.write_text(json.dumps({"queue": str(target), "using_config_path": str(target)}),
                        encoding="utf-8")
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(registry))
    scanner = SequenceQueueReferenceScanner(tmp_path / "products", tmp_path / "products.json", registry)
    dialog = AnalysisModelSelect(str(target), reference_scanner=scanner)
    # Change the draft without the normal autosave notification, so Enter must persist it.
    dialog.select_list.config[0].detail["sample_rate"] = 48000
    dialog.dirty = True
    show(dialog, ui_qapp)
    yield dialog, target
    dispose(dialog, ui_qapp)


@pytest.mark.parametrize("focus", ["analysis_list", "select_list", "new", "import", "save-as",
                                    "top", "up", "down", "bottom", "clear"])
def test_queue_enter_uses_real_save(queue_dialog, focus, enter, ui_qapp):
    dialog, target = queue_dialog
    buttons = dialog.findChildren(QPushButton)
    by_text = {button.text(): button for button in buttons if button.text()}
    moves = [button for button in buttons if not button.text() and button.width() == 30]
    controls = {"analysis_list": dialog.analysis_list, "select_list": dialog.select_list,
                "new": by_text["新建"], "import": by_text["导入"], "save-as": by_text["另存为"]}
    controls.update(zip(("top", "up", "down", "bottom", "clear"), moves))
    clicks = clicks_for(dialog)
    press(dialog, controls[focus], enter, ui_qapp)
    assert clicks == [by_text["保存"]]
    assert json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    assert not dialog.isVisible()


@pytest.fixture(params=["project", "program"])
def product_dialog(request, ui_qapp, tmp_path):
    if request.param == "project":
        manager = make_project_manager(tmp_path)
        prepare_project(manager, tmp_path)
        dialog = ProductTestProjectConfigDialog(manager, queue_editor_callback=lambda path: None)
    else:
        manager = make_program_manager(tmp_path)
        prepare_program(manager)
        dialog = ProductTestProgramConfigDialog(manager, queue_editor_callback=lambda path: None)
    show(dialog, ui_qapp)
    yield dialog, request.param
    dispose(dialog, ui_qapp)


def product_controls(dialog, kind):
    if kind == "project":
        return dialog.condition_table, dialog.project_name_input, dialog.add_condition_btn, dialog.new_project_btn
    return dialog.program_table, dialog.config_combobox.lineEdit(), dialog.add_btn, dialog.new_btn


@pytest.mark.parametrize("focus", ["add", "new", "cancel", "table", "save-as"])
def test_product_enter_uses_existing_save(product_dialog, focus, enter, ui_qapp):
    dialog, kind = product_dialog
    table, name, add, new = product_controls(dialog, kind)
    table.item(0, 1).setText("Edited condition")
    signals = []
    dialog.programs_changed.connect(lambda: signals.append("saved"))
    clicks = clicks_for(dialog)
    controls = {"add": add, "new": new, "cancel": dialog.cancel_btn,
                "table": table, "save-as": dialog.save_as_btn}
    press(dialog, controls[focus], enter, ui_qapp)
    assert clicks == [dialog.save_btn]
    assert signals == ["saved"]
    saved = json.loads(Path(dialog.manager.program_dir, dialog.current_file).read_text(encoding="utf-8"))
    conditions = saved["test_groups"][0]["test_conditions"] if kind == "project" else saved["sub_configs"]
    assert len(conditions) == 1
    assert conditions[0]["condition_name"] == "Edited condition"
    assert dialog.result() == QDialog.Accepted


@pytest.mark.parametrize("focus", ["name", "parameter", "queue"])
def test_product_parameters_keep_values_without_clicking(product_dialog, focus, enter, ui_qapp):
    dialog, kind = product_dialog
    table, name, *_ = product_controls(dialog, kind)
    parameter = dialog.result_root_input if kind == "project" else dialog.close_trigger_input
    combo = table.cellWidget(0, 3).queue_combobox
    widget = {"name": name, "parameter": parameter, "queue": combo}[focus]
    if isinstance(widget, QLineEdit) and not widget.isReadOnly():
        widget.setText("01 04 02 00 00 B9 30" if focus == "parameter" else "Edited name")
    expected = widget.text() if isinstance(widget, QLineEdit) else widget.currentText()
    clicks = clicks_for(dialog)
    before = Path(dialog.manager.program_dir, dialog.current_file).read_bytes()
    press(dialog, widget, enter, ui_qapp)
    assert clicks == []
    assert (widget.text() if isinstance(widget, QLineEdit) else widget.currentText()) == expected
    assert Path(dialog.manager.program_dir, dialog.current_file).read_bytes() == before
    assert dialog.isVisible()


def test_dynamic_table_editor_commits_without_save(product_dialog, enter, ui_qapp):
    dialog, kind = product_dialog
    table, _, add, _ = product_controls(dialog, kind)
    QTest.mouseClick(add, Qt.LeftButton)
    row = table.rowCount() - 1
    item = table.item(row, 1)
    table.setCurrentItem(item)
    table.editItem(item)
    ui_qapp.processEvents()
    editor = dialog.focusWidget()
    assert isinstance(editor, QLineEdit)
    editor.setText("Dynamic condition")
    clicks = clicks_for(dialog)
    press(dialog, editor, enter, ui_qapp)
    assert clicks == []
    assert item.text() == "Dynamic condition"
    data = dialog.collect_project() if kind == "project" else dialog.collect_program()
    conditions = data["test_groups"][0]["test_conditions"] if kind == "project" else data["sub_configs"]
    assert conditions[row]["condition_name"] == "Dynamic condition"
    assert dialog.isVisible()


@pytest.mark.parametrize("focus", ["operation", "settings"])
def test_dynamic_project_buttons_confirm_instead_of_opening_editors(tmp_path, ui_qapp, monkeypatch, enter, focus):
    manager = make_project_manager(tmp_path)
    prepare_project(manager, tmp_path)
    opened = []
    dialog = ProductTestProjectConfigDialog(manager, queue_editor_callback=lambda path: opened.append("queue"))
    monkeypatch.setattr(dialog, "_edit_output_load", lambda cell: opened.append("settings"))
    try:
        show(dialog, ui_qapp)
        QTest.mouseClick(dialog.add_condition_btn, Qt.LeftButton)
        row = dialog.condition_table.rowCount() - 1
        combo, operation = dialog._queue_controls_for_row(row)
        combo.setCurrentIndex(combo.findData("低噪声基础测试"))
        settings = dialog.condition_table.cellWidget(row, 5).settings_button
        ui_qapp.processEvents()
        clicks = clicks_for(dialog)
        press(dialog, operation if focus == "operation" else settings, enter, ui_qapp)
        assert clicks == [dialog.save_btn]
        assert opened == []
        assert dialog.result() == QDialog.Accepted
    finally:
        dispose(dialog, ui_qapp)


def test_product_save_validation_is_preserved(product_dialog, enter, ui_qapp, isolate_prompts):
    dialog, kind = product_dialog
    _, name, add, _ = product_controls(dialog, kind)
    name.clear()
    before = Path(dialog.manager.program_dir, dialog.current_file).read_bytes()
    clicks = clicks_for(dialog)
    press(dialog, add, enter, ui_qapp)
    assert clicks == [dialog.save_btn]
    assert isolate_prompts
    assert dialog.isVisible()
    assert Path(dialog.manager.program_dir, dialog.current_file).read_bytes() == before


@pytest.mark.parametrize("unavailable", ["disabled", "hidden"])
def test_unavailable_product_save_does_not_fall_back(product_dialog, enter, ui_qapp, unavailable):
    dialog, kind = product_dialog
    _, _, add, _ = product_controls(dialog, kind)
    if unavailable == "disabled":
        dialog.save_btn.setEnabled(False)
    else:
        dialog.save_btn.hide()
    clicks = clicks_for(dialog)
    press(dialog, add, enter, ui_qapp)
    assert clicks == []
    assert dialog.isVisible()


@pytest.mark.parametrize("focus", ["cancel", "checkbox"])
def test_copy_conditions_enter_accepts_current_selection(ui_qapp, enter, focus):
    dialog = _CopyConditionsDialog("Source", [(1, "Port B"), (2, "Port C")])
    try:
        dialog._checkboxes[0].setChecked(True)
        show(dialog, ui_qapp)
        buttons = dialog.findChild(QDialogButtonBox)
        clicks = clicks_for(dialog)
        widget = buttons.button(QDialogButtonBox.Cancel) if focus == "cancel" else dialog._checkboxes[1]
        press(dialog, widget, enter, ui_qapp)
        assert clicks == [buttons.button(QDialogButtonBox.Ok)]
        assert dialog.selected_group_indices() == [1]
        assert dialog.result() == QDialog.Accepted
    finally:
        dispose(dialog, ui_qapp)
