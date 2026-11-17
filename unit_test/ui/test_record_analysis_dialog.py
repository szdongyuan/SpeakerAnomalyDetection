"""Native record view intents and lifecycle, with no analysis service."""

from datetime import datetime

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton

from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_request import RecordAnalysisItem
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog


def test_native_parent_navigation_record_and_scroll(qt_app):
    parent = QDialog()
    dialog = RecordAnalysisDialog(parent)
    assert dialog.parent() is parent
    assert dialog.windowTitle() == '记录分析'
    assert dialog.windowModality() == Qt.WindowModal
    assert dialog.testAttribute(Qt.WA_DeleteOnClose)
    items = tuple(RecordAnalysisItem(key, 'FFT', (0,), {}) for key in ('first', 'second'))
    selected = QSignalSpy(dialog.item_selected)
    dialog.set_analysis_items(items)
    assert [dialog.navigation.item(i).data(Qt.UserRole) for i in range(2)] == ['first', 'second']
    assert dialog.navigation.currentRow() == 0 and list(selected[0]) == ['first']
    dialog.navigation.setCurrentRow(1)
    assert list(selected[-1]) == ['second']
    name = 'very-long-file-' * 30 + '.wav'
    dialog.show_record(DailyWavRecord(datetime(2026, 11, 16, 12, 13), 3, name, '正转', 'NG'))
    dialog.show()
    dialog.resize(480, 300)
    content = QLabel('full result')
    content.setMinimumHeight(900)
    dialog.set_content(content)
    qt_app.processEvents()
    assert dialog.file_name.toolTip() == name
    assert dialog.file_name.text() != name
    assert dialog.original_result.text() == '原记录结果：NG'
    assert '仅用于查看' in dialog.read_only_note.text()
    assert dialog.content_scroll.verticalScrollBar().maximum() > 0
    assert not hasattr(dialog, 'return_button')
    assert not any(button.text() == '返回明细' for button in dialog.findChildren(QPushButton))
    second = dialog.navigation.visualItemRect(dialog.navigation.item(1)).center()
    dialog.navigation.setCurrentRow(0)
    QTest.mouseClick(dialog.navigation.viewport(), Qt.LeftButton, pos=second)
    assert list(selected[-1]) == ['second']
    dialog.close()


@pytest.mark.parametrize('method', ['escape', 'system'])
def test_all_close_paths_emit_once(qt_app, method):
    parent = QDialog()
    dialog = RecordAnalysisDialog(parent)
    closed = QSignalSpy(dialog.view_closed)
    dialog.show()
    qt_app.processEvents()
    if method == 'escape':
        QTest.keyClick(dialog, Qt.Key_Escape)
    else:
        dialog.close()
    assert len(closed) == 1


def test_replacing_content_disposes_immediately(qt_app):
    class Content(QLabel):
        disposed = False

        def dispose(self):
            self.disposed = True

    dialog = RecordAnalysisDialog()
    old = Content('old')
    dialog.set_content(old)
    dialog.set_content(QLabel('new'))
    assert old.disposed and old.parent() is None
    dialog.show_status('分析失败', 'cannot read WAV')
    assert dialog.content is None
    assert 'cannot read WAV' in dialog.status_detail.text()
    dialog.close()
