"""Click the actual custom title button before the real main-window close path."""

from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout

from main_window import MainWindow
from ui.sequence import sequence_widget_test_metadata_ops as metadata_ops
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from unit_test.test_test_round_metadata import _MetadataHost


class TitleCloseWindow(MainWindow):
    def __init__(self, sequence):
        QMainWindow.__init__(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.sequence_window = sequence
        self._close_all_subwindows = Mock()
        self.setCentralWidget(sequence)
        layout = QVBoxLayout(sequence)
        layout.addLayout(self.set_title_btn())
        layout.addWidget(sequence.lineedit_type)

    def paintEvent(self, event):
        QMainWindow.paintEvent(self, event)


@pytest.mark.parametrize("pending_analysis", [False, True])
def test_title_x_skips_model_warning_and_preserves_exit_guard(
    ui_qapp, monkeypatch, pending_analysis,
):
    monkeypatch.setattr(metadata_ops.LoadUiConfig, "load_last_recorded_info", lambda *_: None)
    monkeypatch.setattr(metadata_ops, "save_recorded_data_to_json", Mock())
    host = _MetadataHost()
    host._analysis_has_pending_tasks = Mock(return_value=pending_analysis)
    host._save_product_test_progress_before_exit = Mock()
    host._persist_sequence_page_state = Mock()
    window = TitleCloseWindow(host)
    editor = host.lineedit_type
    finish = lambda: SequenceWidgetUiOpsMixin.lineedit_type_lose_focus(host, editor)
    editor.editingFinished.connect(finish)
    real_warning, real_information = QMessageBox.warning, QMessageBox.information

    def dismiss_dialog(method, *args):
        QTimer.singleShot(0, lambda: QApplication.activeModalWidget().accept())
        return method(*args)

    warning = Mock(side_effect=lambda *args: dismiss_dialog(real_warning, *args))
    information = Mock(side_effect=lambda *args: dismiss_dialog(real_information, *args))
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QMessageBox, "information", information)
    try:
        window.show()
        window.activateWindow()
        QTest.qWait(100)
        editor.setReadOnly(False)
        editor.setFocus()
        editor.clear()
        QTest.keyClicks(editor, "d_e")
        assert editor.hasFocus()
        QTest.mouseClick(window.close_btn, Qt.LeftButton)
        QTest.qWait(50)
        warning.assert_not_called()
        assert window.isVisible() is pending_analysis
        if pending_analysis:
            information.assert_called_once()
            assert information.call_args.args[1] == "分析任务未完成"
            host._save_product_test_progress_before_exit.assert_not_called()
            editor.setFocus()
            QTest.keyClick(editor, Qt.Key_Return)
            warning.assert_called_once()
        else:
            information.assert_not_called()
            host._save_product_test_progress_before_exit.assert_called_once()
            window._close_all_subwindows.assert_called_once()
    finally:
        editor.editingFinished.disconnect(finish)
        host._analysis_has_pending_tasks.return_value = False
        window.close()
        host.toolsbar.close()
        window.deleteLater()
        ui_qapp.processEvents()
