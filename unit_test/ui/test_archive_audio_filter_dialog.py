import os
import sqlite3
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from base.audio_record_filter import UNKNOWN, parse_audio_filter_metadata
from base.recording_management import RecordingManager
from consts import error_code
from ui.ai_select_audio_data import SelectAudioDataView
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog, FilterAudioDialog
from unit_test.base.test_audio_record_filter import audio_row, recording_path


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def rows():
    return [
        audio_row("one"),
        audio_row("two", recording_path(
            project="项目B", sample="010", port="端口2", gear="低速档", round_number=10,
        ), label="NG"),
        audio_row("same_name", recording_path(root="D:/results")),
        audio_row("old", "old.wav", label="not_labeled"),
    ]


@pytest.fixture
def dialog(qapp, rows):
    metadata = {row[0]: parse_audio_filter_metadata(row[1]) for row in rows}
    instance = ArchiveAudioFilterDialog(rows, metadata)
    yield instance
    instance.close()


def choose(combo, value):
    index = next(i for i in range(combo.count()) if combo.itemData(i) == value)
    combo.setCurrentIndex(index)


def choices(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_project_selection_restricts_options_and_clears_unavailable_values(dialog):
    choose(dialog.combos["select_port"], "端口1")
    second_project = dialog.metadata_by_id["two"].project_key
    choose(dialog.combos["select_project"], second_project)
    assert choices(dialog.combos["select_sample_number"]) == [None, "010"]
    assert choices(dialog.combos["select_port"]) == [None, "端口2"]
    assert choices(dialog.combos["select_condition"]) == [None, "低速档"]
    assert dialog.combos["select_port"].currentData() is None
    dialog.apply_filters()
    assert dialog.filter_config == {"select_project": second_project}


def test_unknown_selection_and_duplicate_projects_have_distinct_values(dialog):
    projects = dialog.combos["select_project"]
    same_named = [projects.itemText(i) for i in range(projects.count()) if "项目A" in projects.itemText(i)]
    assert len(set(same_named)) == 2
    choose(projects, UNKNOWN)
    assert choices(dialog.combos["select_port"]) == [None, UNKNOWN]
    choose(dialog.combos["select_port"], UNKNOWN)
    dialog.apply_filters()
    assert dialog.filter_config == {"select_project": UNKNOWN, "select_port": UNKNOWN}


def test_apply_reopen_and_reset_preserve_draft_semantics(qapp, rows, dialog):
    choose(dialog.combos["select_sample_number"], "003")
    choose(dialog.combos["select_test_round"], 2)
    dialog.label_boxes["NG"].setChecked(False)
    dialog.rate_boxes[48000].setChecked(False)
    dialog.date_filter_combobox.setEditText("2026-09-13")
    dialog.apply_filters()
    saved = dict(dialog.filter_config)
    reopened = ArchiveAudioFilterDialog(rows, dialog.metadata_by_id, saved)
    assert reopened.combos["select_test_round"].currentData() == 2
    assert reopened.date_filter_combobox.currentText() == "2026-09-13"
    assert saved["select_record_date"] == "2026-09-13"
    assert not reopened.label_boxes["NG"].isChecked()
    assert not reopened.rate_boxes[48000].isChecked()
    reopened.reset_filters()
    assert reopened.filter_config == saved
    assert saved["select_test_round"] == 2
    QTimer.singleShot(0, reopened.reject)
    assert reopened.exec() == (0, {})
    reopened.close()


def test_reset_and_apply_returns_show_all(dialog):
    choose(dialog.combos["select_port"], "端口1")
    dialog.date_filter_combobox.setEditText("2026-09-13")
    dialog.reset_filters()
    assert dialog.date_filter_combobox.currentText() == "ALL"
    QTimer.singleShot(0, dialog.apply_filters)
    assert dialog.exec() == (2, {})


def test_invalid_single_date_or_empty_checkboxes_cannot_apply(dialog, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))
    for invalid in ("2026-02-30", "2026-09", "2026-9-14", ""):
        dialog.date_filter_combobox.setEditText(invalid)
        dialog.apply_filters()
        assert "日期格式错误" in warnings[-1]
        assert dialog.result() != dialog.Accepted
    dialog.reset_filters()
    for box in dialog.rate_boxes.values():
        box.setChecked(False)
    dialog.apply_filters()
    assert "采样率" in warnings[-1]
    dialog.reset_filters()
    for box in dialog.label_boxes.values():
        box.setChecked(False)
    dialog.apply_filters()
    assert "标签" in warnings[-1]


def test_controls_are_visible_and_two_column_layout_fits(dialog, qapp):
    dialog.show()
    qapp.processEvents()
    left = dialog.combos["select_product_model"]
    right = dialog.combos["select_sample_number"]
    assert left.geometry().right() < right.geometry().left()
    for widget in (*dialog.combos.values(), *dialog.rate_boxes.values(),
                   dialog.date_filter_combobox, dialog.apply_button):
        assert widget.isVisible()
        assert dialog.rect().contains(widget.geometry())
        assert widget.height() >= widget.minimumSizeHint().height()
    assert dialog.rate_boxes[44100].text() == "44100 Hz"


def test_single_date_dropdown_reuses_record_dates_and_accepts_typed_date(dialog):
    combo = dialog.date_filter_combobox
    assert combo.isEditable()
    assert [combo.itemText(i) for i in range(combo.count())] == ["ALL", "2026-09-14"]
    combo.setEditText("2026-09-13")
    dialog.apply_filters()
    assert dialog.filter_config == {"select_record_date": "2026-09-13"}


@pytest.fixture
def archive(qapp, rows, monkeypatch):
    monkeypatch.setattr(RecordingManager, "get_record_audio_data", lambda self: (error_code.OK, list(rows)))
    monkeypatch.setattr(RecordingManager, "query_stimulus_name_and_id", lambda self: (error_code.OK, {}))
    instance = ArchiveAudioDataDialog(Mock())
    yield instance
    instance.close()


def test_archive_entry_applies_real_dialog_and_resets_equal_count_filter(archive, monkeypatch):
    assert isinstance(archive.create_filter_dialog({}), ArchiveAudioFilterDialog)
    def execute(dialog):
        choose(dialog.combos["select_port"], "端口1")
        dialog.apply_filters()
        return 1, dialog.filter_config
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", execute)
    archive.on_click_filter_btn()
    assert [row[0] for row in archive.filter_audio_data] == ["one", "same_name"]
    assert archive.model().rowCount() == 2
    archive.model().item(0, 0).setCheckState(Qt.Checked)
    assert archive.select_wave_data["0"][0] == "one"
    archive.show_all_wave()
    assert not archive.select_wave_data
    assert archive.filter_config == {}
    assert archive.model().rowCount() == 4
    # A criterion matching every row must still be cleared by '全部显示'.
    archive.filter_config = {"select_product_model": "型号01"}
    archive.filter_audio_data_at_filter_config(archive.filter_config)
    archive.is_filter_flag = True
    archive.show_all_wave()
    assert not archive.is_filter_flag
    assert archive.filter_config == {}


def test_metadata_is_cached_by_identity_and_refreshed_when_path_changes(archive, monkeypatch):
    import ui.archive_audio_data_dialog as module
    parser = Mock(wraps=parse_audio_filter_metadata)
    monkeypatch.setattr(module, "parse_audio_filter_metadata", parser)
    archive.all_audio_data.reverse()
    archive.filter_audio_data_at_filter_config({"select_port": "端口1"})
    assert parser.call_count == 0
    row = archive.all_audio_data[0]
    archive.all_audio_data[0] = (row[0], recording_path(port="端口3"), *row[2:])
    archive.filter_audio_data_at_filter_config({"select_port": "端口3"})
    assert parser.call_count == 1
    assert [item[0] for item in archive.filter_audio_data] == [row[0]]
    archive.delete_audio_data_with_id([row[0]])
    assert row[0] not in archive._audio_filter_cache


def test_ai_selection_still_uses_legacy_filter_and_hides_unlabeled(archive):
    ai = SelectAudioDataView(Mock(), {}, True)
    dialog = ai.create_filter_dialog({})
    assert isinstance(dialog, FilterAudioDialog)
    assert dialog.select_not_label_check_box.isHidden()
    assert dialog.rotation_speed_combobox is not None
    assert all(row[5] in ("OK", "NG") for row in ai.all_audio_data)
    dialog.close()
    ai.close()


def test_filtering_does_not_open_database_or_scan_for_new_files(archive, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Applying filters must use the existing in-memory records")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    original = list(archive.all_audio_data)
    dialog = archive.create_filter_dialog({})
    choose(dialog.combos["select_sample_number"], "003")
    dialog.apply_filters()
    archive.filter_audio_data_at_filter_config(dialog.filter_config)
    assert archive.all_audio_data == original
    assert len(archive.filter_audio_data) == 2
    dialog.close()


def test_no_match_is_empty_and_cancel_preserves_selection(archive, monkeypatch):
    archive.model().item(0, 0).setCheckState(Qt.Checked)
    selected = dict(archive.select_wave_data)
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", lambda self: (0, {}))
    archive.on_click_filter_btn()
    assert archive.select_wave_data == selected
    assert archive.model().rowCount() == 4
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", lambda self: (
        1, {"select_sample_number": "003", "select_port": "端口2"},
    ))
    archive.on_click_filter_btn()
    assert archive.filter_audio_data == []
    assert archive.model().rowCount() == 0
    assert archive.select_wave_data == {}


def test_legacy_filter_matches_labels_in_label_column_only():
    state = Mock()
    state.all_audio_data = [audio_row("OK", label="not_labeled"), audio_row("real_ok", label="OK")]
    AudioDataManageDialog.filter_audio_data_at_filter_config(state, {"select_labels": ["OK"]})
    assert [row[0] for row in state.filter_audio_data] == ["real_ok"]
