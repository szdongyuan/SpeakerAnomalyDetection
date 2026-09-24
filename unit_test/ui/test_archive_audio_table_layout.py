from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt

from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog


@pytest.fixture
def archive(ui_qapp, monkeypatch):
    rows = [
        (i, f"C:/sample/Motor01_{'long_recording_name_' * 8}{i}.wav",
         "Motor01", 44100, "2026-09-24 10:30:00", "not_labeled", None, None)
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
        for column in (5, 6, 7):
            cell = view.visualRect(archive.model().index(0, column))
            assert cell.left() >= 0
            assert cell.right() < view.viewport().width()
        item = archive.model().item(0, 1)
        assert item.text().endswith(".wav")
        assert item.toolTip() == item.text()
        assert view.fontMetrics().horizontalAdvance(item.text()) > view.columnWidth(1)

    check_visible_columns()
    archive.is_filter_flag = True
    archive.filter_audio_data = [archive.all_audio_data[-1]]
    archive.load_audio_data_to_view()
    ui_qapp.processEvents()
    check_visible_columns()
    assert archive.model().index(0, 7).data(Qt.UserRole) == (
        archive.all_audio_data[-1][0], archive.all_audio_data[-1][1]
    )
    archive.show_all_wave()
    archive.on_clicked_order_btn()
    ui_qapp.processEvents()
    check_visible_columns()
