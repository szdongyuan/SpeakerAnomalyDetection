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
        def __init__(self, *args, **kwargs):
            super().__init__()

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
        self.force_reload_calls = []

    def ensure_config_loaded(self, force_reload=False):
        self.ensure_calls += 1
        self.force_reload_calls.append(force_reload)
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
        self.default_logger = _DummyLogger()
        self.hw_manager = _DummyHardwareManager()
        self.lineedit_type.setText("MODEL")


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


def test_restore_scanner_checkbox_state_accepts_legacy_truthy_string(monkeypatch):
    widget = _DummyWidget()
    save_calls = []

    monkeypatch.setattr(
        barcode_ops.LoadUiConfig,
        "load_last_recorded_info",
        staticmethod(lambda logger: {"scanner_barcode_check": "true"}),
    )
    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", lambda *args: save_calls.append(args), raising=False)

    widget.restore_scanner_checkbox_state()

    assert widget.barcode_scanner_box.isChecked() is True
    assert widget.lineedit_s_or_n.isEnabled() is True
    assert widget.hw_manager.start_calls == 1
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


def test_clicked_scanner_persists_only_checkbox_state(monkeypatch):
    """clicked_scanner must persist ONLY the checkbox flag, never the
    product_model/scanner_barcode fields, so unrelated UI state is preserved."""
    widget = _DummyWidget()
    save_calls = []

    def _capture(*args, **kwargs):
        save_calls.append((args, kwargs))

    monkeypatch.setattr(barcode_ops, "save_recorded_data_to_json", _capture, raising=False)

    widget.barcode_scanner_box.setChecked(True)
    widget.clicked_scanner()
    widget.barcode_scanner_box.setChecked(False)
    widget.clicked_scanner()

    assert save_calls == [
        ((), {"scanner_barcode_check": True}),
        ((), {"scanner_barcode_check": False}),
    ]


def test_save_recorded_data_to_json_merge_preserves_checkbox(tmp_path, monkeypatch):
    """A subsequent save that does NOT mention scanner_barcode_check must
    preserve the previously persisted value (regression test for the bug
    where mode-switch / type-edit lose-focus would overwrite the flag)."""
    from base import save_data

    fake_dir = str(tmp_path) + os.sep
    monkeypatch.setattr(save_data, "DEFAULT_DIR", fake_dir, raising=False)
    os.makedirs(os.path.join(fake_dir, "ui", "ui_config"), exist_ok=True)

    save_data.save_recorded_data_to_json(scanner_barcode_check=True)
    save_data.save_recorded_data_to_json(product_model="MODEL_A", sequence_mode="mark")
    save_data.save_recorded_data_to_json(
        sample_number="SAMPLE-001",
        current_test_round=7,
    )

    import json
    with open(os.path.join(fake_dir, "ui", "ui_config", "recorded_number.json"), "r") as f:
        data = json.load(f)

    assert data["scanner_barcode_check"] is True
    assert data["product_model"] == "MODEL_A"
    assert data["sequence_mode"] == "mark"
    assert data["sample_number"] == "SAMPLE-001"
    assert data["current_test_round"] == 7
