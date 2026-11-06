"""Exercise the real constructor ordering with hardware and persistent UI state isolated."""

import builtins
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget

from ui.sequence import sequence_widget as sequence_module
from ui.sequence.sequence_widget import SequenceWindow
from unit_test.base.test_product_test_config_refresh import make_refresh_manager


@pytest.mark.parametrize("configuration", ["valid", "empty", "invalid", "missing", "registry"])
def test_startup_prepares_once_before_widgets_and_registers_after_ui(
    tmp_path, monkeypatch, configuration,
):
    manager, project, queue_path = make_refresh_manager(tmp_path, condition_count=2, port_count=2)
    filename = project["project_name"] + ".json"
    if configuration == "empty":
        manager.save_registry({"active_file": None, "configs": []})
    elif configuration == "invalid":
        (Path(manager.program_dir) / filename).write_text("{", encoding="utf-8")
    elif configuration == "missing":
        (Path(manager.program_dir) / filename).replace(tmp_path / "moved.json")
    elif configuration == "registry":
        Path(manager.registry_path).write_text("{", encoding="utf-8")
    read = Mock(wraps=manager.load_project)
    monkeypatch.setattr(manager, "load_project", read)
    monkeypatch.setattr(SequenceWindow, "_get_product_program_manager", lambda self: manager)
    monkeypatch.setattr(SequenceWindow, "get_sequence_config_from_registry",
                        lambda self: (str(queue_path), {}))
    monkeypatch.setattr(SequenceWindow, "_load_analysis_window_geometry", lambda self: {})
    for name in ("restore_scanner_checkbox_state",
                 "_restore_last_sequence_mode", "init_lineedit_text", "bind_hw_signals"):
        monkeypatch.setattr(SequenceWindow, name, lambda self: None)
    monkeypatch.setattr(SequenceWindow, "closeEvent",
                        lambda self, event: QWidget.closeEvent(self, event))
    monkeypatch.setattr(sequence_module.LoadUiConfig, "load_last_recorded_info", lambda _: {})
    monkeypatch.setattr(sequence_module, "UnifiedHardwareManager", Mock())
    monkeypatch.setattr(sequence_module.LogManager, "set_log_handler", lambda _: logging.getLogger("startup-test"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    registry_before = Path(manager.registry_path).read_bytes()
    original_open = builtins.open

    def no_statistics_open(file, *args, **kwargs):
        name = str(file).replace("\\", "/")
        assert not name.endswith("/mark_result.json")
        assert "/test_result_log/" not in name
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_statistics_open)
    window = SequenceWindow()
    try:
        expected = {"valid": "failed", "empty": "empty", "invalid": "failed", "missing": "failed", "registry": "failed"}[configuration]
        assert window._product_config_refresh_state == expected
        assert window._pending_initial_product_snapshot is None
        assert window.left_panel.get_station_count() == 0
        assert read.call_count == (0 if configuration in {"empty", "registry"} else 1)
        assert window.product_test_condition_configs == []
        assert not window.player_btn.isEnabled()
        assert window.using_file_combobox.isEnabled()
        if configuration == "valid":
            assert "尚未" in window._product_config_refresh_error
        if configuration == "missing":
            assert window.using_file_combobox.findData(filename) == -1
            assert window.using_file_combobox.currentIndex() == -1
            assert window.using_file_combobox.placeholderText() == "暂无可用配置"
        window.show()
        QApplication.processEvents()
        assert window.left_panel.dashboard_panel.summary_panel.isVisible()
        assert window.left_panel.get_station_count() == 0
        assert len(warnings) == (0 if configuration == "valid" else 1)
        if configuration == "valid":
            assert "尚未" in window.player_btn.toolTip()
            assert "请选择其他" not in window._missing_configuration_message()
        if configuration in {"missing", "empty"}:
            assert warnings[0] == "暂无可用配置。"
        window.hide()
        window.show()
        QApplication.processEvents()
        assert len(warnings) == (0 if configuration == "valid" else 1)
        assert Path(manager.registry_path).read_bytes() == registry_before
    finally:
        window._analysis_process_timer.stop()
        window._round_reset_timer.stop()
        QApplication.instance().removeEventFilter(window)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
