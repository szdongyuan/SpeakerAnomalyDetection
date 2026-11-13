import json
import logging

import pytest
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QWidget

from base import save_data
from ui.sequence import workbench_input_ops as input_ops
from ui.sequence.workbench_input_ops import WorkbenchInputOpsMixin


class Hardware(QObject):
    sig_barcode = pyqtSignal(str)
    sig_trigger = pyqtSignal()
    sig_directional_trigger = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.barcode_source = "hid"
        self.calls = []

    def ensure_config_loaded(self, force_reload=False):
        self.calls.append(("config", force_reload))
        return True

    def start_scanner_listeners(self):
        self.calls.append("start")
        return True

    def stop_scanner_listeners(self):
        self.calls.append("stop")


class Host(WorkbenchInputOpsMixin, QWidget):
    def __init__(self, hardware):
        super().__init__()
        self.hw_manager = hardware
        self.default_logger = logging.getLogger(__name__)
        self.lineedit_type = QLineEdit(self)
        self.lineedit_s_or_n = QLineEdit(self)
        self.barcode_scanner_box = QCheckBox(self)
        self.other_input = QLineEdit(self)
        self.combo = QComboBox(self)
        self.button = QPushButton("focus", self)
        layout = QVBoxLayout(self)
        for widget in (self.lineedit_type, self.lineedit_s_or_n, self.barcode_scanner_box,
                       self.other_input, self.combo, self.button):
            layout.addWidget(widget)
        self.allowed = True
        self.init_workbench_input(lambda: self.allowed)


@pytest.fixture
def host(qt_app, monkeypatch, tmp_path):
    state = tmp_path / "ui/ui_config/recorded_number.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"product_model": "M001", "scanner_barcode_check": True,
                                 "unknown": {"keep": 1}, "sequence_mode": "mark"}))
    monkeypatch.setattr(save_data, "DEFAULT_DIR", str(tmp_path) + "/")
    monkeypatch.setattr(input_ops.LoadUiConfig, "load_last_recorded_info",
                        staticmethod(lambda logger: json.loads(state.read_text())))
    widget = Host(Hardware())
    widget.state_path = state
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close_workbench_input()
    widget.close()


@pytest.mark.parametrize("source", ["hid", "serial"])
def test_hardware_scan_only_updates_sn_and_persists_without_triggering_recording(host, source):
    triggered = []
    host.hw_manager.sig_trigger.connect(lambda: triggered.append(True))
    host.hw_manager.barcode_source = source
    host.bind_hw_signals()  # Rebinding is local and does not duplicate delivery.
    host.hw_manager.sig_barcode.emit(" ABC/123 ")
    assert host.lineedit_s_or_n.text() == "ABC/123"
    assert triggered == []
    saved = json.loads(host.state_path.read_text())
    assert saved["scanner_barcode"] == "ABC/123"
    assert saved["unknown"] == {"keep": 1}
    assert saved["sequence_mode"] == "mark"
    assert saved["product_model"] == "M001"


def test_commit_admission_and_disabled_scanner_do_not_change_sn(host):
    host._commit_barcode("FIRST")
    host.allowed = False
    host.hw_manager.sig_barcode.emit("SECOND")
    assert host.lineedit_s_or_n.text() == "FIRST"
    host.allowed = True
    host.barcode_scanner_box.click()
    host._commit_barcode("THIRD")
    assert host.lineedit_s_or_n.text() == ""
    assert not host.lineedit_s_or_n.isEnabled()
    assert json.loads(host.state_path.read_text())["scanner_barcode_check"] is False


def test_admission_refuses_sn_wedge_input_before_it_can_replace_current_sn(host):
    host._commit_barcode("FIRST")
    host.allowed = False
    QTest.keyClicks(host.lineedit_s_or_n, "SECOND")
    QTest.keyClick(host.lineedit_s_or_n, Qt.Key_Return)
    QTest.qWait(160)
    assert host.lineedit_s_or_n.text() == "FIRST"
    assert json.loads(host.state_path.read_text())["scanner_barcode"] == "FIRST"


def test_commit_deduplicates_for_point_eight_seconds(host, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(input_ops.time, "monotonic", lambda: clock[0])
    host._commit_barcode("FIRST")
    host.lineedit_s_or_n.setText("manual")
    clock[0] = 10.79
    host._commit_barcode("FIRST")
    assert host.lineedit_s_or_n.text() == "manual"
    clock[0] = 10.8
    host._commit_barcode("FIRST")
    assert host.lineedit_s_or_n.text() == "FIRST"


@pytest.mark.parametrize("character", list('\\:*?"<>|') + [" ", "\t", "\n", "\r", "\x00", "\x7f"])
def test_invalid_scan_rejected_and_manual_validation_uses_same_rules(host, monkeypatch, character):
    warnings = []
    monkeypatch.setattr(input_ops.QMessageBox, "warning", lambda *args: warnings.append(args))
    host._commit_barcode("AB/" + character + "123")
    assert host.lineedit_s_or_n.text() == ""
    assert len(warnings) == 1
    host.lineedit_s_or_n.setText("AB/" + character + "123")
    assert host._validate_current_barcode_before_recording() is False


@pytest.mark.parametrize("enter", [True, False])
def test_wedge_global_enter_and_debounce_commit(host, qt_app, enter):
    host.button.setFocus()
    qt_app.processEvents()
    QTest.keyClicks(host.button, "AB/123")
    if enter:
        QTest.keyClick(host.button, Qt.Key_Return)
    else:
        QTest.qWait(160)
    assert host.lineedit_s_or_n.text() == "AB/123"


@pytest.mark.parametrize("enter", [True, False])
def test_sn_wedge_enter_and_fast_input_debounce_overwrite_old_scan(host, enter):
    host._commit_barcode("FIRST")
    QTest.keyClicks(host.lineedit_s_or_n, "SECOND")
    if enter:
        QTest.keyClick(host.lineedit_s_or_n, Qt.Key_Return)
    else:
        QTest.qWait(160)
    assert host.lineedit_s_or_n.text() == "SECOND"
    assert json.loads(host.state_path.read_text())["scanner_barcode"] == "SECOND"


def test_slow_or_short_input_does_not_auto_commit(host):
    host.lineedit_s_or_n.setText("ABC")
    host._barcode_router.on_barcode_debounce_timeout()
    assert "scanner_barcode" not in json.loads(host.state_path.read_text())
    host.lineedit_s_or_n.setText("MANUAL")
    host._barcode_first_char_ts = 1.0
    host._barcode_last_char_ts = 2.0
    host._barcode_router.on_barcode_debounce_timeout()
    assert "scanner_barcode" not in json.loads(host.state_path.read_text())


def test_model_editing_and_input_whitelist_preserve_focus_and_save_on_blur(host, qt_app):
    assert host.lineedit_type.isReadOnly()
    QTest.mouseClick(host.lineedit_type, Qt.LeftButton)
    assert not host.lineedit_type.isReadOnly()
    QTest.keyClicks(host.lineedit_type, "M002")
    host.other_input.setFocus()
    qt_app.processEvents()
    assert host.lineedit_type.isReadOnly()
    assert json.loads(host.state_path.read_text())["product_model"] == "M002"
    QTest.keyClicks(host.other_input, "ABCD")
    QTest.keyClick(host.other_input, Qt.Key_Return)
    assert host.other_input.text() == "ABCD"
    assert host.lineedit_s_or_n.text() == ""
    host.combo.setFocus()
    assert host.eventFilter(host.combo, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a")) is False


def test_hardware_suppresses_duplicate_keystrokes_but_model_remains_editable(host):
    host.hw_manager.sig_barcode.emit("FIRST")
    QTest.keyClicks(host.lineedit_s_or_n, "FIRST")
    assert host.lineedit_s_or_n.text() == "FIRST"
    QTest.mouseClick(host.lineedit_type, Qt.LeftButton)
    QTest.keyClicks(host.lineedit_type, "NEW")
    assert host.lineedit_type.text() == "NEW"


@pytest.mark.parametrize("saved, enabled", [(True, True), ("true", True), ("off", False), (None, False)])
def test_restore_checkbox_does_not_persist_or_start_sensor(host, monkeypatch, saved, enabled):
    monkeypatch.setattr(input_ops.LoadUiConfig, "load_last_recorded_info",
                        staticmethod(lambda logger: {"scanner_barcode_check": saved}))
    before = host.state_path.read_text()
    host.restore_scanner_checkbox_state()
    assert host.barcode_scanner_box.isChecked() is enabled
    assert host.state_path.read_text() == before
    assert host.hw_manager.calls[-1] == ("start" if enabled else "stop")


def test_close_is_idempotent_and_disconnects_only_owned_barcode_slot(host):
    class ExternalService:
        def close(self):
            pytest.fail("Input cleanup must not close external workbench services")

        stop = shutdown = close

    host.recording_service = ExternalService()
    host._raw_audio_csv_service = ExternalService()
    host._analysis_service = ExternalService()
    others = []
    host.hw_manager.sig_barcode.connect(others.append)
    host.hw_manager.sig_trigger.connect(lambda: others.append("trigger"))
    host.hw_manager.sig_directional_trigger.connect(others.append)
    before = host.hw_manager.calls.count("stop")
    host.close_workbench_input()
    host.close_workbench_input()
    host.bind_hw_signals()
    host.hw_manager.sig_barcode.emit("AFTER_CLOSE")
    host.hw_manager.sig_trigger.emit()
    host.hw_manager.sig_directional_trigger.emit("forward")
    assert others == ["AFTER_CLOSE", "trigger", "forward"]
    assert host.lineedit_s_or_n.text() == ""
    assert host.hw_manager.calls.count("stop") == before + 1
    assert not host._barcode_debounce_timer.isActive()

def test_model_validation_does_not_reenter_during_modal_warning(host, monkeypatch):
    warnings = []
    def warning(*args):
        warnings.append(args)
        if len(warnings) == 1:
            host.lineedit_type.editingFinished.emit()
    monkeypatch.setattr(input_ops.QMessageBox, 'warning', warning)
    host.lineedit_type.setText('INVALID/MODEL')
    host.lineedit_type.editingFinished.emit()
    assert len(warnings) == 1
    host.lineedit_type.setText('VALID-MODEL')
    host.lineedit_type.editingFinished.emit()
    assert host.lineedit_type.isReadOnly()
    assert json.loads(host.state_path.read_text())['product_model'] == 'VALID-MODEL'


@pytest.mark.parametrize("model", ["", "test_prodect", "A.1", "A()", "COM1", "A" * 81])
def test_invalid_product_model_keeps_text_and_does_not_persist(host, monkeypatch, model):
    from unittest.mock import Mock
    warning, saved = Mock(), Mock()
    monkeypatch.setattr(input_ops.QMessageBox, "warning", warning)
    monkeypatch.setattr(input_ops, "save_recorded_data_to_json", saved)
    host.lineedit_type.setText(model)
    host.lineedit_type.setReadOnly(False)
    host.lineedit_type.editingFinished.emit()
    assert host.lineedit_type.text() == model
    assert not host.lineedit_type.isReadOnly()
    warning.assert_called_once()
    saved.assert_not_called()


def test_product_model_can_be_typed_in_stages_then_normalized_and_saved(host, monkeypatch):
    from unittest.mock import Mock
    warning, saved = Mock(), Mock()
    monkeypatch.setattr(input_ops.QMessageBox, "warning", warning)
    monkeypatch.setattr(input_ops, "save_recorded_data_to_json", saved)
    editor = host.lineedit_type
    editor.setReadOnly(False)
    editor.clear()
    QTest.keyClicks(editor, "ABC(V2")
    warning.assert_not_called()
    QTest.keyClick(editor, Qt.Key_Return)
    warning.assert_called_once()
    assert editor.text() == "ABC(V2"
    editor.deselect()
    editor.setCursorPosition(len(editor.text()))
    QTest.keyClicks(editor, ")")
    QTest.keyClick(editor, Qt.Key_Return)
    assert editor.text() == "ABC(V2)"
    assert editor.isReadOnly()
    saved.assert_called_once_with(product_model="ABC(V2)", scanner_barcode=host.lineedit_s_or_n.text())

def test_busy_sn_keeps_tab_focus_navigation(host, qt_app):
    QWidget.setTabOrder(host.lineedit_s_or_n, host.other_input)
    host._commit_barcode("FIRST")
    host.allowed = False
    assert qt_app.focusWidget() is host.lineedit_s_or_n
    QTest.keyClick(host.lineedit_s_or_n, Qt.Key_Tab)
    assert qt_app.focusWidget() is host.other_input
    QTest.keyClick(host.other_input, Qt.Key_Backtab)
    assert qt_app.focusWidget() is host.lineedit_s_or_n
    assert host.lineedit_s_or_n.text() == "FIRST"


@pytest.mark.parametrize("key,modifiers,position,selection", [
    (Qt.Key_Home, Qt.NoModifier, 0, ""),
    (Qt.Key_Left, Qt.ShiftModifier, 4, "T"),
    (Qt.Key_A, Qt.ControlModifier, 5, "FIRST"),
])
def test_busy_sn_keeps_cursor_and_selection_navigation(host, key, modifiers, position, selection):
    host._commit_barcode("FIRST")
    host.lineedit_s_or_n.deselect()
    host.lineedit_s_or_n.setCursorPosition(5)
    host.allowed = False
    QTest.keyClick(host.lineedit_s_or_n, key, modifiers)
    assert host.lineedit_s_or_n.cursorPosition() == position
    assert host.lineedit_s_or_n.selectedText() == selection
    assert host.lineedit_s_or_n.text() == "FIRST"


@pytest.mark.parametrize("key,modifiers", [
    (Qt.Key_Backspace, Qt.NoModifier), (Qt.Key_Delete, Qt.NoModifier),
    (Qt.Key_Delete, Qt.ShiftModifier), (Qt.Key_Insert, Qt.ShiftModifier),
    (Qt.Key_X, Qt.ControlModifier), (Qt.Key_V, Qt.ControlModifier),
    (Qt.Key_Z, Qt.ControlModifier), (Qt.Key_Y, Qt.ControlModifier),
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.NoModifier),
])
def test_busy_sn_consumes_modification_and_commit_keys_without_dispatch(host, key, modifiers):
    host._commit_barcode("FIRST")
    host.allowed = False
    # Test the filter boundary so cut/paste cannot touch the user's clipboard.
    event = QKeyEvent(QEvent.KeyPress, key, modifiers)
    assert host.eventFilter(host.lineedit_s_or_n, event) is True
    assert host.lineedit_s_or_n.text() == "FIRST"
    assert json.loads(host.state_path.read_text())["scanner_barcode"] == "FIRST"


@pytest.mark.parametrize("key,modifiers", [
    (Qt.Key_C, Qt.ControlModifier), (Qt.Key_Insert, Qt.ControlModifier),
])
def test_busy_sn_allows_copy_at_filter_boundary_without_using_clipboard(host, key, modifiers):
    host._commit_barcode("FIRST")
    host.allowed = False
    assert host.eventFilter(host.lineedit_s_or_n, QKeyEvent(QEvent.KeyPress, key, modifiers)) is False
