from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QItemSelectionModel, Qt
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QAbstractItemView

from base.audio_label_repository import AudioLabelUpdate
from base.recording_management import RecordingManager
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog


@pytest.fixture
def archive(ui_qapp, tmp_path, monkeypatch):
    rows = [
        (str(i), str(tmp_path / f"recording-{i}.wav"), "型号一", 44100,
         "2026-11-13", "not_labeled" if i % 2 == 0 else "OK", f"ABC/{i}/XY", None)
        for i in range(50)
    ]
    query = Mock(return_value=(error_code.OK, list(rows)))
    monkeypatch.setattr(RecordingManager, "get_record_audio_data", query)
    monkeypatch.setattr(RecordingManager, "query_stimulus_name_and_id",
                        lambda self: (error_code.OK, {}))
    playback = Mock()
    playback.is_audio_playing.return_value = False

    def start(path):
        playback.is_audio_playing.return_value = True
        playback.get_current_playing_file.return_value = path
        return error_code.OK, "playing"

    def stop():
        playback.is_audio_playing.return_value = False

    playback.start_audio_playback.side_effect = start
    playback.stop_audio_playback.side_effect = stop
    monkeypatch.setattr("ui.archive_audio_data_dialog.PlaybackController", lambda: playback)
    window = ArchiveAudioDataDialog(Mock())
    window._play_col_debounce_ms = 0
    monkeypatch.setattr(window, "_get_audio_total_seconds", lambda path: 600.0)
    window.show()
    ui_qapp.processEvents()
    query.reset_mock()
    playback.reset_mock()
    try:
        yield window, playback, rows, ui_qapp, query
    finally:
        window._stop_playback_if_needed()
        window.close()
        window.deleteLater()
        ui_qapp.processEvents()


def apply(window, row, label):
    window.apply_audio_label_update(AudioLabelUpdate(row[0], row[1], label))


def filtered(window, labels):
    window.filter_config = {"select_labels": labels}
    window.is_filter_flag = True
    window.filter_audio_data_at_filter_config(window.filter_config)
    window.load_audio_data_to_view()
    window.set_select_wave_num_text(0)


def visible_ids(window):
    return [window.model().index(i, window._analysis_btn_col).data(Qt.UserRole)[0]
            for i in range(window.model().rowCount())]


def retained_items(window):
    return {identifier: [window.model().item(i, col) for col in range(9)]
            for i, identifier in enumerate(visible_ids(window))}


def assert_retained(window, before):
    for i, identifier in enumerate(visible_ids(window)):
        if identifier in before:
            assert [window.model().item(i, col) for col in range(9)] == before[identifier]


@pytest.mark.parametrize("reload", [False, True])
def test_analysis_action_text_and_identity_survive_label_update_after_loading(archive, reload):
    window, playback, rows, app, query = archive
    if reload:
        window.load_audio_data_to_view()
    action = window.model().item(0, window._analysis_btn_col)
    assert action.text() == "查看"
    assert action.data(Qt.UserRole) == rows[0][:2]

    apply(window, rows[0], "NG")
    window._refresh_play_button_states()

    assert window.model().item(0, window._analysis_btn_col) is action
    assert action.text() == "查看"
    assert action.data(Qt.UserRole) == rows[0][:2]


def test_unfiltered_update_changes_only_label_and_refreshes_cached_selection(archive):
    window, playback, rows, app, query = archive
    window.model().item(12, 0).setCheckState(Qt.Checked)
    window.data_view.setCurrentIndex(window.model().index(12, 3))
    window.data_view.verticalScrollBar().setValue(10)
    scroll = window.data_view.verticalScrollBar().value()
    before = retained_items(window)
    reset = QSignalSpy(window.model().modelReset)
    inserted = QSignalSpy(window.model().rowsInserted)
    removed = QSignalSpy(window.model().rowsRemoved)

    apply(window, rows[12], "NG")

    assert window.all_audio_data[12] == (*rows[12][:5], "NG", *rows[12][6:])
    assert window.filter_audio_data[12] == window.all_audio_data[12]
    assert window.select_wave_data == {"12": window.all_audio_data[12]}
    assert window.all_audio_data[:12] == rows[:12]
    assert window.all_audio_data[13:] == rows[13:]
    assert window.model().item(12, 4).text() == "NG"
    assert_retained(window, before)
    assert window.data_view.currentIndex() == window.model().index(12, 3)
    assert window.data_view.verticalScrollBar().value() == scroll
    assert not reset and not inserted and not removed
    query.assert_not_called()
    playback.start_audio_playback.assert_not_called()
    playback.stop_audio_playback.assert_not_called()


def test_descending_filtered_update_uses_identity_without_reordering(archive):
    window, playback, rows, app, query = archive
    window.on_clicked_order_btn()
    filtered(window, ["OK", "not_labeled"])
    config = dict(window.filter_config)
    before = visible_ids(window)
    apply(window, rows[7], "not_labeled")
    assert visible_ids(window) == before == [row[0] for row in reversed(rows)]
    assert window._is_desc_order
    assert window.filter_config == config
    assert window.is_filter_flag
    assert window.model().item(42, 4).text() == "not_labeled"
    assert window.filter_audio_data[42][0] == "7"
    query.assert_not_called()


@pytest.mark.parametrize("entering", [False, True])
def test_filter_membership_change_preserves_items_checks_selection_and_scroll(archive, entering):
    window, playback, rows, app, query = archive
    filtered(window, ["not_labeled"])
    window.data_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    window.data_view.setCurrentIndex(window.model().index(14, 3))
    selection = window.data_view.selectionModel()
    selection.select(window.model().index(12, 2), QItemSelectionModel.Select)
    window.model().item(14, 0).setCheckState(Qt.Checked)
    window.model().item(0, 0).setCheckState(Qt.Checked)
    window.data_view.verticalScrollBar().setValue(200)
    app.processEvents()
    top = window.data_view.rowAt(0)
    anchor = visible_ids(window)[top]
    offset = window.data_view.visualRect(window.model().index(top, 0)).top()
    before = retained_items(window)
    removed = QSignalSpy(window.model().rowsRemoved)
    inserted = QSignalSpy(window.model().rowsInserted)
    reset = QSignalSpy(window.model().modelReset)

    apply(window, rows[1 if entering else 0], "not_labeled" if entering else "OK")
    app.processEvents()

    ids = visible_ids(window)
    assert ids == [row[0] for row in window.filter_audio_data]
    assert len(ids) == (26 if entering else 24)
    assert_retained(window, before)
    assert len(inserted) == int(entering)
    assert len(removed) == int(not entering)
    assert not reset
    assert window.data_view.currentIndex() == window.model().index(ids.index("28"), 3)
    assert {(ids[index.row()], index.column()) for index in selection.selectedIndexes()} == {
        ("28", 3), ("24", 2),
    }
    expected_checked = {"0", "28"} if entering else {"28"}
    assert {row[0] for row in window.select_wave_data.values()} == expected_checked
    assert window.select_wave_data == {
        str(i): row for i, row in enumerate(window.filter_audio_data) if row[0] in expected_checked
    }
    assert not window.all_selected_checkbox.isChecked()
    assert not window.all_select_flag
    assert window.select_data_num.text() == f"已选择 {len(expected_checked)} 个文件, 共 {len(ids)} 个文件"
    new_top = window.data_view.rowAt(0)
    assert ids[new_top] == anchor
    assert window.data_view.visualRect(window.model().index(new_top, 0)).top() == offset
    assert window.data_view.is_send_signal
    query.assert_not_called()

    if entering:
        new_row = ids.index("1")
        assert window.model().item(new_row, 0).isCheckable()
        assert window.model().item(new_row, 1).toolTip() == "recording-1.wav"
        assert window.model().item(new_row, 3).toolTip() == "ABC/1/XY"
        assert window.model().item(new_row, 7).text() == "播放"
        assert window.model().item(new_row, 8).data(Qt.UserRole) == ("1", rows[1][1])


@pytest.mark.parametrize("change", ["own_label", "earlier_leaves", "earlier_enters"])
def test_still_visible_playback_keeps_timing_and_follows_record(archive, change):
    window, playback, rows, app, query = archive
    filtered(window, ["not_labeled", "NG"] if change == "own_label" else ["not_labeled"])
    window._on_play_button_clicked(10)
    before = retained_items(window)
    state = (window._current_playing_file, window._playback_started_monotonic,
             window._current_playing_total_sec, window._playback_poll_timer.isActive(),
             window._playback_poll_timer.timerId())
    playback.reset_mock()
    if change == "own_label":
        apply(window, rows[20], "NG")
    elif change == "earlier_leaves":
        apply(window, rows[0], "OK")
    else:
        apply(window, rows[1], "not_labeled")
    assert window._current_playing_row == visible_ids(window).index("20")
    assert (window._current_playing_file, window._playback_started_monotonic,
            window._current_playing_total_sec, window._playback_poll_timer.isActive(),
            window._playback_poll_timer.timerId()) == state
    assert window.model().item(window._current_playing_row, 7).text() == "停止"
    assert_retained(window, before)
    playback.start_audio_playback.assert_not_called()
    playback.stop_audio_playback.assert_not_called()


def test_playing_record_filtered_out_stops_once_and_clears_selected_tuple(archive):
    window, playback, rows, app, query = archive
    filtered(window, ["not_labeled"])
    window.set_all_checkboxes_checked([0], True)
    window._on_play_button_clicked(0)
    before = retained_items(window)
    playback.reset_mock()
    apply(window, rows[0], "NG")
    playback.stop_audio_playback.assert_called_once()
    playback.start_audio_playback.assert_not_called()
    assert window._current_playing_row is None
    assert not window._playback_poll_timer.isActive()
    assert window._playback_started_monotonic is None
    assert all(row[0] != "0" for row in window.select_wave_data.values())
    assert window.all_selected_checkbox.isChecked()
    assert window.all_select_flag
    assert_retained(window, before)


@pytest.mark.parametrize("identifier,path", [("unknown", "same"), ("0", "different")])
def test_unknown_identity_or_mismatched_path_is_ignored(archive, identifier, path):
    window, playback, rows, app, query = archive
    before = retained_items(window)
    window.apply_audio_label_update(AudioLabelUpdate(
        identifier, rows[0][1] if path == "same" else rows[1][1], "NG",
    ))
    assert window.all_audio_data == rows
    assert visible_ids(window) == [row[0] for row in rows]
    assert_retained(window, before)
    query.assert_not_called()


def test_normalized_notification_path_matches_cached_path(archive):
    window, playback, rows, app, query = archive
    path = rows[0][1].replace("\\", "/")
    parent, name = path.rsplit("/", 1)
    window.apply_audio_label_update(AudioLabelUpdate("0", f"{parent}/./{name}", "NG"))
    assert window.all_audio_data[0][5] == "NG"


def test_relative_cached_path_is_resolved_from_application_root(archive, monkeypatch, tmp_path):
    from pathlib import Path

    window, playback, rows, app, query = archive
    record = rows[0]
    window.all_audio_data[0] = (record[0], "results/recording.wav", *record[2:])
    window.load_audio_data_to_view()
    monkeypatch.chdir(tmp_path)
    path = str(Path(DEFAULT_DIR) / "results" / "recording.wav")
    window.apply_audio_label_update(AudioLabelUpdate("0", path, "NG"))
    assert window.all_audio_data[0][1:2] == ("results/recording.wav",)
    assert window.model().item(0, 4).text() == "NG"


def test_final_filtered_row_leaving_clears_selection_and_all_select(archive):
    window, playback, rows, app, query = archive
    window.all_audio_data[0] = (*rows[0][:5], "NG", *rows[0][6:])
    filtered(window, ["NG"])
    window.model().item(0, 0).setCheckState(Qt.Checked)
    window.data_view.setCurrentIndex(window.model().index(0, 4))
    apply(window, rows[0], "OK")
    assert window.model().rowCount() == 0
    assert window.filter_audio_data == []
    assert window.select_wave_data == {}
    assert not window.data_view.currentIndex().isValid()
    assert not window.all_select_flag
    assert not window.all_selected_checkbox.isChecked()
    assert window.select_data_num.text() == "已选择 0 个文件, 共 0 个文件"
    apply(window, rows[0], "NG")
    assert visible_ids(window) == ["0"]
    assert window.model().item(0, 4).text() == "NG"
    assert not window.select_wave_data
    assert window.data_view.is_send_signal


def test_hidden_cached_record_update_refreshes_cache_without_adding_ineligible_row(archive):
    window, playback, rows, app, query = archive
    filtered(window, ["not_labeled"])
    before = retained_items(window)
    apply(window, rows[1], "NG")
    assert window.all_audio_data[1][5] == "NG"
    assert "1" not in visible_ids(window)
    assert len(window.filter_audio_data) == 25
    assert_retained(window, before)
    query.assert_not_called()
