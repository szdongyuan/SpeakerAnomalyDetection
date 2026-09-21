"""Serial settings use temporary files and a fake transport, never real ports."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget

from base.load_config import LoadUiConfig
from ui import serial_discrete_input_config_dialog as dialog_module
from ui.sequence import sequence_widget_serial_trigger_ops as serial_ops


Dialog = dialog_module.SerialDiscreteInputConfigDialog


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "serial_discrete_input.json"
    config = LoadUiConfig.get_default_serial_discrete_input_config()
    config["serial_settings"]["port"] = "COM4"
    config["simulator_sequence"] = {"steps": [{"hex": "A5 5A 01 01 0D 0A"}]}
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(path))
    return path


def set_ports(monkeypatch, names):
    ports = [SimpleNamespace(device=name, description="Virtual port") for name in names]
    monkeypatch.setattr(dialog_module, "list_ports", SimpleNamespace(comports=lambda: ports))


@pytest.mark.parametrize("ports", [["COM2", "COM4"], ["COM2"], []])
def test_open_preserves_saved_port_even_when_not_enumerated(ui_qapp, monkeypatch, config_file, ports):
    set_ports(monkeypatch, ports)
    _, config = LoadUiConfig.load_serial_discrete_input_config()
    dialog = Dialog(config)
    try:
        assert dialog.port_combobox.currentText().startswith("COM4")
        assert dialog._build_config() == config
    finally:
        dialog.close()


@pytest.mark.parametrize("typed_port", ["COM4", "COM16", ""])
def test_manual_port_text_overrides_old_item_data(ui_qapp, monkeypatch, typed_port):
    set_ports(monkeypatch, ["COM2"])
    dialog = Dialog({"serial_settings": {"port": "COM2"}})
    try:
        dialog.port_combobox.setEditText(typed_port)
        if typed_port:
            assert dialog._build_config()["serial_settings"]["port"] == typed_port
        else:
            with pytest.raises(ValueError):
                dialog._build_config()
    finally:
        dialog.close()


class Hardware:
    def __init__(self, config):
        self.serial_config = deepcopy(config)
        self.running = bool(config["enabled"])
        self.serial_worker = SimpleNamespace(isRunning=lambda: self.running)
        self.events = []

    def get_serial_discrete_input_status(self):
        return {"running": self.running, "connected": self.running}

    def start_serial_discrete_input_listener(self, config, full_frame_candidates):
        self.serial_config = deepcopy(config)
        self.running = True
        self.events.append(("start", deepcopy(config)))
        return {"ok": True}

    def stop_serial_discrete_input_listener(self, *, for_reconfiguration=False):
        self.running = False
        self.events.append(("stop", None))

    def test_serial_discrete_input_connection(self, config):
        self.events.append(("test", deepcopy(config)))
        return {"ok": True, "raw_hex": "A5 5A 01 01 0D 0A"}


class Host(QWidget, serial_ops.SequenceWidgetSerialTriggerOpsMixin):
    def __init__(self, config):
        super().__init__()
        self._serial_trigger_config = deepcopy(config)
        self.hw_manager = Hardware(config)

    def _serial_full_frame_candidates(self, config):
        return ("A5 5A 01 01 0D 0A",)

    def _sync_product_progress_after_trigger_switch(self):
        pass

    def on_serial_trigger_status_changed(self, status):
        self.last_status = status

    def update_player_btn_is_paused(self):
        pass


@pytest.mark.parametrize("action", ["save", "disable", "cancel", "test_cancel", "save_failure"])
@pytest.mark.parametrize("polling", [False, True])
def test_dialog_json_and_listener_lifecycle(ui_qapp, monkeypatch, config_file, action, polling):
    set_ports(monkeypatch, ["COM2", "COM4"])
    _, initial = LoadUiConfig.load_serial_discrete_input_config()
    initial["enabled"] = action in ("disable", "test_cancel", "save_failure")
    initial["polling_settings"]["enabled"] = not polling
    assert LoadUiConfig.save_serial_discrete_input_config(initial)
    original_bytes = config_file.read_bytes()
    host = Host(initial)
    windows = []
    warnings = []
    monkeypatch.setattr(dialog_module.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    monkeypatch.setattr(Dialog, "_show_test_result_popup", lambda *args: None)
    if action == "save_failure":
        monkeypatch.setattr(LoadUiConfig, "save_serial_discrete_input_config", lambda config: False)

    def create_dialog(*args, **kwargs):
        dialog = Dialog(*args, **kwargs)
        windows.append(dialog)

        def interact():
            dialog.enabled_checkbox.setChecked(action == "save")
            dialog.communication_mode_combobox.setCurrentIndex(1 if polling else 0)
            if polling:
                dialog.polling_interval_spinbox.setValue(1200)
            dialog.port_combobox.setEditText("COM4" if action == "test_cancel" else "COM16")
            dialog.baudrate_combobox.setCurrentText("115200")
            dialog.device_model_lineedit.setText("Fixture")
            if action == "test_cancel":
                dialog._on_test_btn_clicked()
            if action in ("cancel", "test_cancel"):
                dialog.close()
            else:
                dialog._on_ok_btn_clicked()

        QTimer.singleShot(0, interact)
        return dialog

    monkeypatch.setattr(serial_ops, "SerialDiscreteInputConfigDialog", create_dialog)
    try:
        host.on_serial_trigger_btn_clicked()
        _, saved = LoadUiConfig.load_serial_discrete_input_config()
        if action in ("cancel", "test_cancel", "save_failure"):
            assert config_file.read_bytes() == original_bytes
            assert host._serial_trigger_config == initial
            assert host.hw_manager.running == initial["enabled"]
            if action == "test_cancel":
                assert [event[0] for event in host.hw_manager.events] == ["stop", "test", "start"]
                assert host.hw_manager.serial_config == initial
                assert host.hw_manager.events[1][1]["polling_settings"]["enabled"] is polling
                assert host.hw_manager.events[1][1]["polling_settings"]["interval_ms"] == (1200 if polling else 50)
            else:
                assert host.hw_manager.events == []
        else:
            assert saved == host._serial_trigger_config == windows[0]._build_config()
            assert saved["serial_settings"]["port"] == "COM16"
            assert saved["serial_settings"]["baudrate"] == 115200
            assert saved["device_model"] == "Fixture"
            assert saved["polling_settings"]["enabled"] is polling
            assert saved["polling_settings"]["query_command_hex"] == initial["polling_settings"]["query_command_hex"]
            assert saved["polling_settings"]["interval_ms"] == (1200 if polling else 50)
            assert saved["simulator_sequence"] == initial["simulator_sequence"]
            assert host.hw_manager.running == saved["enabled"]
            if action == "save":
                assert host.hw_manager.serial_config == saved
            reopened = Dialog(saved)
            windows.append(reopened)
            assert reopened._build_config() == saved
        assert bool(warnings) == (action == "save_failure")
        assert host._serial_trigger_config_dialog_open is False
    finally:
        for dialog in windows:
            dialog.close()
        host.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_mode_opens_from_json_and_switches_without_mutating_input(ui_qapp, monkeypatch, enabled):
    set_ports(monkeypatch, ["COM4"])
    config = LoadUiConfig.get_default_serial_discrete_input_config()
    config["polling_settings"]["enabled"] = enabled
    dialog = Dialog(config)
    try:
        assert dialog.communication_mode_combobox.currentData() is enabled
        dialog.communication_mode_combobox.setCurrentIndex(0 if enabled else 1)
        assert dialog._build_config()["polling_settings"]["enabled"] is not enabled
        assert config["polling_settings"]["enabled"] is enabled
        assert "polling_settings.enabled" in Dialog.EDITABLE_PATHS
    finally:
        dialog.close()


def test_interval_control_range_and_mode_state(ui_qapp, monkeypatch):
    set_ports(monkeypatch, ["COM4"])
    config = LoadUiConfig.get_default_serial_discrete_input_config()
    dialog = Dialog(config)
    try:
        control = dialog.polling_interval_spinbox
        assert control.minimum() == 1 and control.maximum() == 1200
        assert control.value() == 50
        assert not control.isEnabled()
        dialog.communication_mode_combobox.setCurrentIndex(1)
        assert control.isEnabled()
        control.setValue(1201)
        assert control.value() == 1200
        control.setValue(0)
        assert control.value() == 1
        dialog.communication_mode_combobox.setCurrentIndex(0)
        assert not control.isEnabled()
        assert dialog._build_config()["polling_settings"]["interval_ms"] == 1
    finally:
        dialog.close()


@pytest.mark.parametrize("action", ["save", "test"])
def test_enabling_polling_without_query_keeps_dialog_open(ui_qapp, monkeypatch, action):
    set_ports(monkeypatch, ["COM4"])
    config = LoadUiConfig.get_default_serial_discrete_input_config()
    config["polling_settings"]["query_command_hex"] = ""
    warnings, calls = [], []
    monkeypatch.setattr(dialog_module.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = Dialog(config, test_connection_callback=calls.append)
    try:
        dialog.communication_mode_combobox.setCurrentIndex(1)
        if action == "save":
            dialog._on_ok_btn_clicked()
        else:
            dialog._on_test_btn_clicked()
        assert dialog._dialog_action is None
        assert warnings == ["启用轮询时 query_command_hex 不能为空"]
        assert calls == []
    finally:
        dialog.close()
