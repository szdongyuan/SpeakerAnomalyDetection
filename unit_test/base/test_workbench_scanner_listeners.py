import logging

import pytest
from PyQt5.QtCore import QObject

from base.unified_hid_device_manager import UnifiedHardwareManager


@pytest.fixture
def manager(qt_app, monkeypatch):
    manager = UnifiedHardwareManager.__new__(UnifiedHardwareManager)
    QObject.__init__(manager)
    manager.logger = logging.getLogger(__name__)
    manager.hotkey_registered = True
    manager.calls = []
    for name in ("ensure_config_loaded", "_start_hid_scanner", "_stop_hid_scanner_internal",
                 "_start_serial_barcode_worker", "_stop_serial_barcode_worker",
                 "_start_sensor_hotkey_if_enabled", "start_serial_discrete_input_listener"):
        monkeypatch.setattr(manager, name, lambda name=name: manager.calls.append(name))
    return manager


@pytest.mark.parametrize("source, expected", [
    ("hid", ["_start_hid_scanner"]),
    ("wedge", ["_stop_hid_scanner_internal"]),
    ("serial", ["_stop_hid_scanner_internal", "_start_serial_barcode_worker"]),
])
def test_scanner_only_start_dispatches_without_sensor_or_serial_test_trigger(manager, source, expected):
    manager.barcode_source = source
    assert manager.start_scanner_listeners() is True
    assert manager.calls == ["ensure_config_loaded"] + expected


def test_combined_entry_still_starts_sensor(manager):
    manager.barcode_source = "wedge"
    assert manager.start_scanner_and_sensor_listeners() is True
    assert manager.calls[-1] == "_start_sensor_hotkey_if_enabled"


def test_scanner_only_stop_preserves_hotkey_and_other_consumers(manager):
    received = []
    manager.sig_trigger.connect(lambda: received.append("trigger"))
    manager.sig_directional_trigger.connect(received.append)
    manager.stop_scanner_listeners()
    manager.sig_trigger.emit()
    manager.sig_directional_trigger.emit("forward")
    assert manager.calls == ["_stop_hid_scanner_internal", "_stop_serial_barcode_worker"]
    assert manager.hotkey_registered is True
    assert received == ["trigger", "forward"]


def test_scanner_failure_keeps_existing_diagnostic_and_keyboard_fallback(manager, monkeypatch, caplog):
    def fail():
        raise OSError("scanner unavailable")
    monkeypatch.setattr(manager, "_start_scanner_by_source", fail)
    with caplog.at_level(logging.ERROR):
        assert manager.start_scanner_listeners() is False
    assert "scanner unavailable" in caplog.text
    assert "降级" in caplog.text
