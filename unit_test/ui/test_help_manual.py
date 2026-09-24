from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAction, QWidget

import main_window as main_window_module


class MenuHarness(QWidget):
    init_menu = main_window_module.MainWindow.init_menu

    def on_help_manual(self):
        return main_window_module.MainWindow.on_help_manual(self)

    def __init__(self):
        super().__init__()
        action_slots = {
            "function_action_product_test_program": "on_product_test_program_config",
            "function_action_test_sequence": "analysis_model_select",
            "function_audio_manager": "on_audio_manager_init",
            "function_action_report_export": "on_analysis_report_export",
            "function_action_exit": "on_window_close",
            "hardware_action_selection": "on_hardware_window_init",
            "hardware_action_calibration": "on_calibration_window_init",
            "user_action_switch_account": "on_login_window_init",
            "user_action_add_account": "on_add_account_window_init",
            "user_action_change_pwd": "on_change_pwd_window_init",
        }
        for action_name, slot_name in action_slots.items():
            slot = Mock()
            setattr(self, slot_name, slot)
            action = QAction(action_name, self)
            action.triggered.connect(slot)
            setattr(self, action_name, action)


@pytest.fixture
def harness(ui_qapp):
    widget = MenuHarness()
    yield widget
    widget.close()
    widget.deleteLater()
    ui_qapp.processEvents()


@pytest.fixture
def manual_environment(tmp_path, monkeypatch):
    project = tmp_path / "源码 目录"
    deployed = tmp_path / "迁移后的 程序"
    other = tmp_path / "启动目录"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setattr(main_window_module, "__file__", str(project / "main_window.py"))
    runtime = SimpleNamespace(
        frozen=False,
        executable=str(deployed / "主程序.exe"),
        _MEIPASS=str(tmp_path / "临时解包目录"),
    )
    monkeypatch.setattr(main_window_module, "sys", runtime)
    paths = []
    for root in (project, deployed):
        path = root / "ui/ui_config/希听异音测试系统用户使用手册.html"
        path.parent.mkdir(parents=True)
        path.write_text("<html><body>手册</body></html>", encoding="utf-8")
        paths.append(path)
    opened = Mock(return_value=True)
    warnings = Mock()
    # QtGui 已被现有主窗口导入；补丁在新增 import 前同样可用。
    monkeypatch.setattr("PyQt5.QtGui.QDesktopServices.openUrl", opened)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", warnings)
    return runtime, paths, opened, warnings


@pytest.mark.parametrize("frozen", [False, True])
def test_help_click_opens_manual_after_directory_change(
    harness, ui_qapp, manual_environment, frozen
):
    runtime, paths, opened, warnings = manual_environment
    runtime.frozen = frozen
    expected = paths[int(frozen)]
    menu = harness.init_menu()
    menu.setParent(harness)
    assert [action.text() for action in menu.actions()] == ["功能", "硬件", "用户", "帮助"]
    action = menu.actions()[-1]
    assert action.menu() is None
    harness.resize(640, 80)
    menu.resize(640, 40)
    harness.show()
    menu.show()
    ui_qapp.processEvents()
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())
    ui_qapp.processEvents()
    opened.assert_called_once()
    url = opened.call_args.args[0]
    assert url.isLocalFile()
    assert Path(url.toLocalFile()) == expected
    warnings.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_missing_or_directory_manual_warns_without_opening(harness, manual_environment, kind):
    runtime, paths, opened, warnings = manual_environment
    path = paths[0]
    path.unlink()
    if kind == "directory":
        path.mkdir()
    harness.on_help_manual()
    opened.assert_not_called()
    warnings.assert_called_once()
    assert warnings.call_args.args[1] == "帮助"
    assert "未找到用户使用手册" in warnings.call_args.args[2]
    assert str(path) in warnings.call_args.args[2]


def test_open_failure_explains_recovery(harness, manual_environment):
    runtime, paths, opened, warnings = manual_environment
    opened.return_value = False
    harness.on_help_manual()
    opened.assert_called_once()
    warnings.assert_called_once()
    message = warnings.call_args.args[2]
    assert "默认浏览器" in message
    assert "手动打开" in message
    assert str(paths[0]) in message


def test_rebuilding_menu_does_not_duplicate_open_requests(harness, manual_environment):
    runtime, paths, opened, warnings = manual_environment
    for count in (1, 2):
        menu = harness.init_menu()
        menu.setParent(harness)
        menu.actions()[-1].trigger()
        assert opened.call_count == count
    warnings.assert_not_called()


def test_manual_file_check_failure_warns_with_reason(
    harness, manual_environment, monkeypatch
):
    runtime, paths, opened, warnings = manual_environment
    with monkeypatch.context() as local_patch:
        local_patch.setattr(Path, "is_file", Mock(side_effect=PermissionError("拒绝访问")))
        harness.on_help_manual()
    opened.assert_not_called()
    warnings.assert_called_once()
    assert warnings.call_args.args[1] == "帮助"
    message = warnings.call_args.args[2]
    assert str(paths[0]) in message
    assert "拒绝访问" in message
