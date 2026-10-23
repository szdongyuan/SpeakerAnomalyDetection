import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from unit_test.base.test_audio_analysis_result_source import recording
from unit_test.ui.test_archive_audio_analysis_dialog import wait_until, close_view


def test_current_row_after_filter_order_and_all_display(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    first, _, _ = recording(tmp_path)
    second, _, _ = recording(tmp_path, project="项目乙")
    rows = [(i, str(wav), "型号一", 44100, "2026-09-15", "OK", f"ABC/{i}/XY", None)
            for i, wav in enumerate((first, second))]
    monkeypatch.setattr(RecordingManager, "get_record_audio_data", lambda self: (error_code.OK, rows))
    monkeypatch.setattr(RecordingManager, "query_stimulus_name_and_id", lambda self: (error_code.OK, {}))
    window = ArchiveAudioDataDialog(Mock())
    window.show()
    try:
        window.is_filter_flag = True
        window.filter_audio_data = [rows[1]]
        window.load_audio_data_to_view()
        window.set_all_checkboxes_checked([0], True)
        before = dict(window.select_wave_data)
        index = window.model().index(0, window._analysis_btn_col)
        assert index.data(Qt.UserRole) == (1, str(second))
        assert window.model().index(0, 3).data() == "ABC/1/XY"
        window.data_view.scrollTo(index)
        app.processEvents()
        QTest.mouseClick(window.data_view.viewport(), Qt.LeftButton,
                         pos=window.data_view.visualRect(index).center())
        dialog = window._analysis_dialog
        assert dialog.wav_path == str(second)
        assert window.select_wave_data == before
        wait_until(lambda: dialog.results is not None and dialog.loader._thread is None)
        close_view(dialog)
        assert window._analysis_dialog is None
        window.show_all_wave()
        window.on_clicked_order_btn()
        displayed = [window.model().index(row, window._analysis_btn_col).data(Qt.UserRole)[1]
                     for row in range(window.model().rowCount())]
        assert displayed == [str(second), str(first)]
        assert window.model().index(0, window._play_btn_col).data() == "播放"
        assert window.model().index(0, 3).data() == "ABC/1/XY"

        playback = Mock()
        playback.is_audio_playing.return_value = False

        def start_playback(path):
            playback.is_audio_playing.return_value = True
            playback.get_current_playing_file.return_value = path
            return error_code.OK, "playing"

        def stop_playback():
            playback.is_audio_playing.return_value = False

        playback.start_audio_playback.side_effect = start_playback
        playback.stop_audio_playback.side_effect = stop_playback
        window.playback_controller = playback
        window._play_col_debounce_ms = 0
        monkeypatch.setattr(window, "_get_audio_total_seconds", lambda _path: 1.0)
        play_index = window.model().index(0, window._play_btn_col)
        window.data_view.scrollTo(play_index)
        app.processEvents()
        QTest.mouseClick(window.data_view.viewport(), Qt.LeftButton,
                         pos=window.data_view.visualRect(play_index).center())
        playback.start_audio_playback.assert_called_once_with(str(second))
        assert play_index.data() == "停止"
        QTest.mouseClick(window.data_view.viewport(), Qt.LeftButton,
                         pos=window.data_view.visualRect(play_index).center())
        playback.stop_audio_playback.assert_called_once()
        assert play_index.data() == "播放"
    finally:
        window.close()
        app.processEvents()
