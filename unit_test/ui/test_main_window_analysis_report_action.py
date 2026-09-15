import logging
from types import SimpleNamespace

import main_window as main_window_module
import ui.analysis_report_export_dialog as report_dialog_module


def test_report_export_action_is_available_to_operator(ui_qapp, monkeypatch):
    monkeypatch.setattr(main_window_module.LogManager, "set_log_handler", logging.getLogger)
    monkeypatch.setattr(main_window_module, "restore_or_default", lambda **kwargs: (None, None, [], []))
    monkeypatch.setattr(main_window_module.MainWindow, "init_ui", lambda self: None)
    monkeypatch.setattr(main_window_module.MainWindow, "_init_ve_hardware_runtime", lambda self: None)
    bridge = SimpleNamespace(shutdown=lambda: None)

    window = main_window_module.MainWindow(recording_bridge=bridge)

    assert window.function_action_report_export.text() == "报告导出"
    assert window.function_action_report_export in window.widget_list_operator
    window.access_lvl = "Operator"
    window.on_access_lvl_changed()
    assert window.function_action_report_export.isEnabled()


def test_report_export_action_opens_dialog(ui_qapp, monkeypatch):
    opened = []

    class _Dialog:
        def __init__(self, parent):
            opened.append(parent)

        @staticmethod
        def exec():
            return 0

    monkeypatch.setattr(report_dialog_module, "AnalysisReportExportDialog", _Dialog)
    host = object()

    main_window_module.MainWindow.on_analysis_report_export(host)

    assert opened == [host]
