"""Serial settings use temporary files and a fake transport, never real ports."""

import json
from types import SimpleNamespace

import pytest

from base.load_config import LoadUiConfig
from ui import serial_discrete_input_config_dialog as dialog_module


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
