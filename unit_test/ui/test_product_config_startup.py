"""Exercise the real constructor ordering with hardware and persistent UI state isolated."""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QWidget

from ui.sequence import sequence_widget as sequence_module
from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequencement_count_board import SequenceCountBoard
from unit_test.base.test_product_test_config_refresh import make_refresh_manager


@pytest.mark.parametrize("configuration", ["valid", "empty", "invalid"])
def test_startup_prepares_once_before_widgets_and_registers_after_ui(
    tmp_path, monkeypatch, configuration,
):
    manager, _, queue_path = make_refresh_manager(tmp_path, condition_count=2, port_count=2)
    if configuration == "empty":
        manager.save_registry({"active_file": None, "configs": []})
    elif configuration == "invalid":
        (Path(manager.program_dir) / "A.json").write_text("{", encoding="utf-8")
    read = Mock(wraps=manager.load_project)
    monkeypatch.setattr(manager, "load_project", read)
    monkeypatch.setattr(SequenceWindow, "_get_product_program_manager", lambda self: manager)
    monkeypatch.setattr(SequenceWindow, "get_sequence_config_from_registry",
                        lambda self: (str(queue_path), {}))
    monkeypatch.setattr(SequenceWindow, "_load_analysis_window_geometry", lambda self: {})
    for name in ("init_result_files", "reset_statistics_on_startup", "restore_scanner_checkbox_state",
                 "_restore_last_sequence_mode", "init_lineedit_text", "bind_hw_signals"):
        monkeypatch.setattr(SequenceWindow, name, lambda self: None)
    monkeypatch.setattr(SequenceWindow, "closeEvent", QWidget.closeEvent)
    for name in ("set_test_text", "set_mark_text"):
        monkeypatch.setattr(SequenceCountBoard, name, lambda self: None)
    monkeypatch.setattr(sequence_module.LoadUiConfig, "load_last_recorded_info", lambda _: {})
    monkeypatch.setattr(sequence_module, "UnifiedHardwareManager", Mock())
    monkeypatch.setattr(sequence_module.LogManager, "set_log_handler", lambda _: logging.getLogger("startup-test"))
    window = SequenceWindow()
    expected = {"valid": "ready", "empty": "empty", "invalid": "failed"}[configuration]
    assert window._product_config_refresh_state == expected
    assert window._pending_initial_product_snapshot is None
    assert read.call_count == (0 if configuration == "empty" else 1)
    if configuration == "valid":
        assert len(window.product_test_condition_configs) == 4
        snapshot = window._applied_product_snapshot
        key = window.product_test_condition_configs[0]["key"]
        window.left_panel.set_condition_result(key, "NG")
        window.on_product_test_program_updated()
        assert window._applied_product_snapshot is snapshot
        assert window.left_panel.result_panel.rows[key]["result"] == "NG"
    else:
        assert window.product_test_condition_configs == []
        assert not window.player_btn.isEnabled()
        assert window.using_file_combobox.isEnabled()
    window._analysis_process_timer.stop()
    window._round_reset_timer.stop()
    QApplication.instance().removeEventFilter(window)
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
