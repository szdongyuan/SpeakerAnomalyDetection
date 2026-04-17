import logging
import os
import sys
import types

import pytest
from PyQt5.QtWidgets import QApplication, QCheckBox, QLineEdit

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence import sequence_widget_barcode_ops as barcode_ops
from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin


class _DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class _DummyHardwareManager:
    def __init__(self):
        self.ensure_calls = 0
        self.start_calls = 0
        self.stop_calls = 0

    def ensure_config_loaded(self):
        self.ensure_calls += 1
        return True

    def start_scanner_and_sensor_listeners(self):
        self.start_calls += 1
        return True

    def stop_scanner_and_sensor_listeners(self):
        self.stop_calls += 1


class _DummyWidget(SequenceWidgetBarcodeOpsMixin):
    def __init__(self):
        self.barcode_scanner_box = QCheckBox()
        self.lineedit_s_or_n = QLineEdit()
        self.lineedit_type = QLineEdit()
        self.lineedit_count = QLineEdit()
        self.default_logger = _DummyLogger()
        self.hw_manager = _DummyHardwareManager()
        self.lineedit_type.setText("MODEL")
        self.lineedit_count.setText("7")


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def test_restore_scanner_checkbox_state_uses_saved_flag(monkeypatch):
    widget = _DummyWidget()
    saved_payload = {"scanner_barcode_check": True}
    save_calls = []

    monkeypatch.setattr(barcode_ops.LoadUiConfig, "load_last_recorded_info", staticmethod(lambda logger: saved_payload))
    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", lambda *args: save_calls.append(args), raising=False)

    widget.restore_scanner_checkbox_state()

    assert widget.barcode_scanner_box.isChecked() is True
    assert widget.lineedit_s_or_n.isEnabled() is True
    assert widget.hw_manager.start_calls == 1
    assert save_calls == []


def test_restore_scanner_checkbox_state_defaults_to_disabled(monkeypatch):
    widget = _DummyWidget()
    widget.barcode_scanner_box.setChecked(True)
    widget.lineedit_s_or_n.setEnabled(True)
    widget.lineedit_s_or_n.setText("SN001")
    save_calls = []

    monkeypatch.setattr(
        barcode_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: {"scanner_barcode_check": "invalid"}),
    )
    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", lambda *args: save_calls.append(args), raising=False)

    widget.restore_scanner_checkbox_state()

    assert widget.barcode_scanner_box.isChecked() is False
    assert widget.lineedit_s_or_n.isEnabled() is False
    assert widget.lineedit_s_or_n.text() == ""
    assert widget.hw_manager.stop_calls == 1
    assert save_calls == []


@pytest.mark.parametrize("saved_payload", [None, [1], "invalid"])
def test_restore_scanner_checkbox_state_handles_missing_or_malformed_payload(monkeypatch, saved_payload):
    widget = _DummyWidget()
    widget.barcode_scanner_box.setChecked(True)
    widget.lineedit_s_or_n.setEnabled(True)
    widget.lineedit_s_or_n.setText("SN001")
    save_calls = []

    monkeypatch.setattr(
        barcode_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: saved_payload),
    )
    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", lambda *args: save_calls.append(args), raising=False)

    widget.restore_scanner_checkbox_state()

    assert widget.barcode_scanner_box.isChecked() is False
    assert widget.lineedit_s_or_n.isEnabled() is False
    assert widget.lineedit_s_or_n.text() == ""
    assert widget.hw_manager.stop_calls == 1
    assert save_calls == []


def test_clicked_scanner_persists_checkbox_state(monkeypatch):
    widget = _DummyWidget()
    save_calls = []

    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", lambda *args: save_calls.append(args), raising=False)

    widget.barcode_scanner_box.setChecked(True)
    widget.clicked_scanner()
    widget.barcode_scanner_box.setChecked(False)
    widget.clicked_scanner()

    assert save_calls == [
        ("MODEL", "7", "", True),
        ("MODEL", "7", "", False),
    ]
