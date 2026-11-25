from copy import deepcopy
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QInputDialog, QMessageBox

from base.load_config import LoadUiConfig
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from unit_test.test_product_test_project_config import make_manager, make_project
from unit_test.test_serial_product_ports import A, IDLE


@pytest.mark.parametrize("single_condition", [True, False])
@pytest.mark.parametrize("serial_enabled", [False, True])
def test_save_button_accepts_missing_idle_and_explains_runtime_limit(
    ui_qapp, tmp_path, monkeypatch, single_condition, serial_enabled,
):
    serial_path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(serial_path))
    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": serial_enabled, "port_switch_idle_code": ""})
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    project_path = Path(manager.program_dir, filename)
    before = project_path.read_bytes()
    draft = deepcopy(project)
    if single_condition:
        draft["test_groups"] = draft["test_groups"][:1]
    for group in draft["test_groups"]:
        group["test_conditions"][0]["trigger_state"] = A
    warnings = []
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: warnings.append(message))
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, message: messages.append(message))
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog._show_project(draft, filename)
        dialog.save_btn.click()
        assert warnings == []
        conflict = serial_enabled and not single_condition
        assert ("无法使用串口自动触发" in messages[-1]) == conflict
        assert "尚不可运行" not in messages[-1]
        assert manager.validate_project(draft, filename)["is_usable"] == (not conflict)
        assert project_path.read_bytes() != before
        assert dialog.result() == dialog.Accepted
        assert manager.load_project(filename)[1]["test_groups"][0]["test_conditions"][0]["trigger_state"] == A

        assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": IDLE})
        dialog.save_btn.click()
        assert warnings == []
        assert "无法使用串口自动触发" not in messages[-1]
        assert dialog.result() == dialog.Accepted
        assert manager.load_project(filename)[1]["test_groups"][0]["test_conditions"][0]["trigger_state"] == A
    finally:
        dialog._dirty = False
        dialog.close()


@pytest.mark.parametrize("idle", ["", IDLE], ids=["conflicting-ports", "valid-idle"])
@pytest.mark.parametrize("serial_enabled", [False, True])
def test_save_as_reports_serial_conflict_on_saved_copy(ui_qapp, tmp_path, monkeypatch, idle, serial_enabled):
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path",
                        lambda: str(tmp_path / "serial.json"))
    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": serial_enabled, "port_switch_idle_code": idle})
    manager = make_manager(tmp_path)
    draft = make_project(tmp_path, project_name="方案 A")
    ok, original = manager.save_project(None, draft)
    assert ok
    for group in draft["test_groups"]:
        group["test_conditions"][0]["trigger_state"] = A
    messages = []
    def accept_name(widget):
        widget.setTextValue("方案 B")
        return QInputDialog.Accepted
    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    monkeypatch.setattr(QMessageBox, "information", lambda _, title, text: messages.append(text))
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog._show_project(draft, original)
        dialog.save_as_btn.click()
        expected = "已另存为“方案 B”，当前仍在编辑“方案 A”。"
        if serial_enabled and not idle:
            expected = ("已另存为“方案 B”。无法使用串口自动触发：USB-C输出口的末档与USB-A输出口的首档状态码相同。"
                        "请在【功能 → 产品测试程序配置】中核对并修改状态码。")
        assert messages == [expected]
        assert dialog.current_file == manager.load_registry()["active_file"] == original
        copy = manager.load_project("方案 B.json")[1]
        assert copy["test_groups"][1]["test_conditions"][0]["trigger_state"] == A
        assert manager.validate_project(copy, "方案 B.json")["is_usable"] == (not serial_enabled or bool(idle))
        assert manager.load_project(original)[1]["test_groups"][1]["test_conditions"][0]["trigger_state"] == ""
    finally:
        dialog._dirty = False
        dialog.close()
