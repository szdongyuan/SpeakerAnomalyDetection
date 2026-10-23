from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtWidgets import QToolTip

from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog


@pytest.fixture
def archive(ui_qapp, monkeypatch):
    barcodes = ["LONG/" * 40, "ABC/123/XY", "SN001", None, ""]
    rows = [
        (i, f"C:/sample/Motor01_{'long_recording_name_' * 8}{i}.wav",
         "Motor01", 44100, "2026-09-24 10:30:00", "not_labeled", barcodes[i % 5], None)
        for i in range(40)
    ]
    monkeypatch.setattr(
        RecordingManager, "get_record_audio_data", lambda self: (error_code.OK, rows)
    )
    monkeypatch.setattr(
        RecordingManager, "query_stimulus_name_and_id", lambda self: (error_code.OK, {})
    )
    dialog = ArchiveAudioDataDialog(Mock())
    dialog.show()
    ui_qapp.processEvents()
    yield dialog
    dialog.close()


@pytest.mark.parametrize("width", [1150, 1100, 1050, 920])
def test_long_names_leave_metadata_and_actions_visible(archive, ui_qapp, width):
    archive.resize(width, 520)
    ui_qapp.processEvents()

    def check_visible_columns():
        view = archive.data_view
        assert view.horizontalScrollBar().maximum() == 0
        for column in (3, 6, 7, 8):
            cell = view.visualRect(archive.model().index(0, column))
            assert cell.left() >= 0
            assert cell.right() < view.viewport().width()
        item = archive.model().item(0, 1)
        assert item.text().endswith(".wav")
        assert item.toolTip() == item.text()
        assert view.fontMetrics().horizontalAdvance(item.text()) > view.columnWidth(1)
        rows = archive.filter_audio_data if archive.is_filter_flag else archive.all_audio_data
        for row, record in enumerate(rows):
            barcode_item = archive.model().item(row, 3)
            assert barcode_item.text() == (record[6] or "")
            assert barcode_item.toolTip().replace("\n", "") == (record[6] or "")
            assert barcode_item.data(Qt.ForegroundRole) is None
            assert not (barcode_item.textAlignment() & Qt.AlignHCenter)
            assert archive.model().index(row, 8).data(Qt.UserRole) == record[:2]

    check_visible_columns()
    assert archive.model().headerData(3, Qt.Horizontal) == "条形码"
    barcode_item = archive.model().item(0, 3)
    assert archive.data_view.fontMetrics().horizontalAdvance(barcode_item.text()) > (
        archive.data_view.columnWidth(3)
    )
    archive.is_filter_flag = True
    archive.filter_audio_data = [archive.all_audio_data[-1]]
    archive.load_audio_data_to_view()
    ui_qapp.processEvents()
    check_visible_columns()
    assert archive.model().index(0, 8).data(Qt.UserRole) == (
        archive.all_audio_data[-1][0], archive.all_audio_data[-1][1]
    )
    archive.show_all_wave()
    archive.on_clicked_order_btn()
    ui_qapp.processEvents()
    check_visible_columns()


def test_long_barcode_tooltip_shows_complete_value_within_screen(archive, ui_qapp):
    view = archive.data_view
    item = archive.model().item(0, 3)
    pos = view.visualRect(item.index()).center()
    ui_qapp.sendEvent(view.viewport(), QHelpEvent(
        QEvent.ToolTip, pos, view.viewport().mapToGlobal(pos),
    ))
    ui_qapp.processEvents()
    try:
        tooltip = next(widget for widget in ui_qapp.topLevelWidgets()
                       if widget.objectName() == "qtooltip_label")
        assert tooltip.isVisible()
        assert tooltip.text().replace("\n", "") == item.text()
        assert tooltip.width() <= tooltip.screen().availableGeometry().width()
    finally:
        QToolTip.hideText()
