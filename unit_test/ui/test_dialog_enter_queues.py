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
from ui.confirmation_message_box import ConfirmationMessageBox
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_project_config_dialog import (
    ProductTestProjectConfigDialog,
)
from unit_test.test_product_test_project_config_dialog import (
    make_manager as make_project_manager, prepare_project,
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
    monkeypatch.setattr(ConfirmationMessageBox, "question", lambda *a: QMessageBox.No)
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


@pytest.mark.parametrize("focus", ["add", "new", "cancel"])
def test_product_enter_uses_existing_save(tmp_path, ui_qapp, enter, focus):
    manager = make_project_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager=manager)
    try:
        show(dialog, ui_qapp)
        controls = {"add": dialog.condition_editor.add_button,
                    "new": dialog.new_project_btn, "cancel": dialog.cancel_btn}
        clicks = clicks_for(dialog)
        press(dialog, controls[focus], enter, ui_qapp)
        assert clicks == [dialog.save_btn]
        assert not dialog.isVisible()
    finally:
        dialog._dirty = False
        dispose(dialog, ui_qapp)
