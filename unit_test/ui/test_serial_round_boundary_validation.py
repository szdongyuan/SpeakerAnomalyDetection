from copy import deepcopy
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from base.load_config import LoadUiConfig
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from unit_test.test_product_test_project_config import make_manager, make_project
from unit_test.test_serial_product_ports import A, IDLE


def test_save_button_rejects_same_round_boundary_until_idle_configured(
    ui_qapp, tmp_path, monkeypatch,
):
    serial_path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(serial_path))
    assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": ""})
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    project_path = Path(manager.program_dir, filename)
    before = project_path.read_bytes()
    draft = deepcopy(project)
    draft["test_groups"] = draft["test_groups"][:1]
    draft["test_groups"][0]["test_conditions"][0]["trigger_state"] = A
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: warnings.append(message))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog._show_project(draft, filename)
        dialog.save_btn.click()
        assert len(warnings) == 1
        assert "本轮末档与下一轮首档状态码相同" in warnings[0]
        assert project_path.read_bytes() == before
        assert dialog.result() != dialog.Accepted

        assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": IDLE})
        dialog.save_btn.click()
        assert len(warnings) == 1
        assert dialog.result() == dialog.Accepted
        assert manager.load_project(filename)[1]["test_groups"][0]["test_conditions"][0]["trigger_state"] == A
    finally:
        dialog._dirty = False
        dialog.close()
