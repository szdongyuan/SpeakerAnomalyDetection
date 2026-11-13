"""Custom title close uses the current application drain without model validation."""
from time import monotonic
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMessageBox

from unit_test.ui.test_workbench_recording_startup import startup


@pytest.mark.parametrize('pending_analysis', [False, True])
def test_title_x_skips_model_warning_and_preserves_exit_drain(startup, ui_qapp, monkeypatch, pending_analysis):
    window = startup.launch(lambda _: ('Engineer', 'test'))
    startup.analysis_service.allow_close = not pending_analysis
    warning = Mock()
    monkeypatch.setattr(QMessageBox, 'warning', warning)
    editor = window.sequence_window.lineedit_type
    try:
        window.activateWindow()
        QTest.qWait(100)
        editor.setReadOnly(False)
        editor.setFocus()
        editor.clear()
        QTest.keyClicks(editor, 'd_e')
        assert editor.hasFocus()
        QTest.mouseClick(window.close_btn, Qt.LeftButton)
        QTest.qWait(150)
        warning.assert_not_called()
        assert window.isVisible() is pending_analysis
        assert startup.analysis_service.shutdown_calls == 1
        assert window.sequence_window._workbench_input_closed is (not pending_analysis)
    finally:
        startup.analysis_service.allow_close = True
        window.close()
        deadline = monotonic() + 5
        while not getattr(window, '_exit_cleanup_complete', False) and monotonic() < deadline:
            ui_qapp.processEvents()
            QTest.qWait(10)
        assert window._exit_cleanup_complete
