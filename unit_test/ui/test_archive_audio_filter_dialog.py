import builtins
import os
import sqlite3
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox

from base.audio_record_filter import parse_archive_audio_filter_metadata
from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog, FilterAudioDialog
from unit_test.base.test_audio_record_filter import audio_row
from unit_test.base.test_current_audio_record_filter import current_recording_path


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def rows():
    return [
        audio_row("one", current_recording_path()),
        audio_row("two", current_recording_path(
            project="项目B", station="工位2", condition="反转",
        ), label="NG"),
        audio_row("same_name", current_recording_path(root="D:/results")),
        audio_row("unlabeled", current_recording_path(station="工位10", condition="反转"),
                  label="not_labeled"),
    ]


@pytest.fixture
def dialog(qapp, rows):
    metadata = {row[0]: parse_archive_audio_filter_metadata(row[1]) for row in rows}
    instance = ArchiveAudioFilterDialog(rows, metadata)
    yield instance
    instance.close()
    instance.deleteLater()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)


def choose(combo, value):
    index = next(i for i in range(combo.count()) if combo.itemData(i) == value)
    combo.setCurrentIndex(index)


def choices(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_current_controls_apply_only_station_and_condition(dialog):
    assert set(dialog.combos) == {
        "select_project", "select_product_model", "select_station", "select_condition",
    }
    for key, text in (("select_station", "全部工位"), ("select_condition", "全部工况")):
        assert dialog.combos[key].itemText(0) == text
        assert not dialog.combos[key].isEditable()
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert {"工位", "工况"} <= labels
    assert not {"样本编号", "测试轮次", "端口", "档位"} & labels
    choose(dialog.combos["select_station"], "工位1")
    choose(dialog.combos["select_condition"], "正转")
    dialog.apply_filters()
    assert dialog.filter_config == {"select_station": "工位1", "select_condition": "正转"}


def test_project_selection_restricts_options_and_clears_unavailable_values(dialog):
    choose(dialog.combos["select_station"], "工位1")
    choose(dialog.combos["select_condition"], "正转")
    choose(dialog.combos["select_product_model"], "型号01")
    second_project = dialog.metadata_by_id["two"].project_key
    choose(dialog.combos["select_project"], second_project)
    assert choices(dialog.combos["select_product_model"]) == [None, "型号01"]
    assert dialog.combos["select_product_model"].currentData() == "型号01"
    assert choices(dialog.combos["select_station"]) == [None, "工位2"]
    assert choices(dialog.combos["select_condition"]) == [None, "反转"]
    assert dialog.combos["select_station"].currentData() is None
    assert dialog.combos["select_condition"].currentData() is None
    dialog.apply_filters()
    assert dialog.filter_config == {"select_project": second_project, "select_product_model": "型号01"}


def test_names_are_unique_naturally_sorted_and_duplicate_projects_distinct(dialog):
    projects = dialog.combos["select_project"]
    same_named = [projects.itemText(i) for i in range(projects.count()) if "项目A" in projects.itemText(i)]
    assert len(set(same_named)) == 2
    assert len(set(choices(projects)[1:])) == 3
    assert choices(dialog.combos["select_station"]) == [None, "工位1", "工位2", "工位10"]
    choose(dialog.combos["select_station"], "工位1")
    assert choices(dialog.combos["select_condition"]) == [None, "反转", "正转"]
    choose(dialog.combos["select_condition"], "反转")
    assert choices(dialog.combos["select_station"]) == [None, "工位1", "工位2", "工位10"]
    choose(projects, dialog.metadata_by_id["one"].project_key)
    assert dialog.combos["select_station"].currentData() == "工位1"
    assert dialog.combos["select_condition"].currentData() == "反转"


def test_empty_records_offer_only_unrestricted_options(qapp):
    dialog = ArchiveAudioFilterDialog([], {})
    assert set(dialog.combos) == {
        "select_project", "select_product_model", "select_station", "select_condition",
    }
    assert all(choices(combo) == [None] for combo in dialog.combos.values())
    dialog.apply_filters()
    assert dialog.filter_config == {}
    dialog.close()


def test_apply_reopen_and_reset_preserve_draft_semantics(qapp, rows, dialog):
    choose(dialog.combos["select_station"], "工位1")
    choose(dialog.combos["select_condition"], "正转")
    dialog.label_boxes["NG"].setChecked(False)
    dialog.sample_rate_combobox.setEditText("44100")
    dialog.date_filter_combobox.setEditText("2026-09-13")
    dialog.barcode_lineedit.setText("ABC/123/XY")
    dialog.apply_filters()
    saved = dict(dialog.filter_config)
    reopened = ArchiveAudioFilterDialog(rows, dialog.metadata_by_id, saved)
    assert reopened.combos["select_station"].currentData() == "工位1"
    assert reopened.combos["select_condition"].currentData() == "正转"
    assert reopened.date_filter_combobox.currentText() == "2026-09-13"
    assert saved["select_record_date"] == "2026-09-13"
    assert not reopened.label_boxes["NG"].isChecked()
    assert reopened.sample_rate_combobox.currentText() == "44100"
    assert reopened.barcode_lineedit.text() == "ABC/123/XY"
    assert saved["select_barcode"] == "ABC/123/XY"
    reopened.reset_filters()
    assert reopened.barcode_lineedit.text() == ""
    assert reopened.filter_config == saved
    assert saved["select_station"] == "工位1"
    QTimer.singleShot(0, reopened.reject)
    assert reopened.exec() == (0, {})
    reopened.close()


def test_reset_and_apply_returns_show_all(dialog):
    choose(dialog.combos["select_station"], "工位1")
    dialog.date_filter_combobox.setEditText("2026-09-13")
    dialog.sample_rate_combobox.setEditText("96000")
    dialog.barcode_lineedit.setText("ABC/123/XY")
    dialog.reset_filters()
    assert dialog.barcode_lineedit.text() == ""
    assert dialog.date_filter_combobox.currentText() == "ALL"
    assert dialog.sample_rate_combobox.currentText() == "ALL"
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
    dialog.sample_rate_combobox.setEditText("")
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
    right = dialog.combos["select_station"]
    assert left.geometry().right() < right.geometry().left()
    assert left.geometry().top() == right.geometry().top()
    condition = dialog.combos["select_condition"]
    assert condition.geometry().top() <= dialog.barcode_lineedit.geometry().bottom()
    assert dialog.barcode_lineedit.geometry().top() <= condition.geometry().bottom()
    assert condition.geometry().right() < dialog.barcode_lineedit.geometry().left()
    grid = dialog.layout().itemAt(0).layout()
    assert grid.getItemPosition(grid.indexOf(condition)) == (5, 0, 1, 1)
    assert grid.getItemPosition(grid.indexOf(dialog.barcode_lineedit)) == (5, 1, 1, 1)
    barcode_label = next(label for label in dialog.findChildren(QLabel) if label.text() == "条形码")
    assert grid.getItemPosition(grid.indexOf(barcode_label)) == (4, 1, 1, 1)
    for widget in (*dialog.combos.values(), dialog.sample_rate_combobox,
                   dialog.date_filter_combobox, dialog.barcode_lineedit, dialog.apply_button,
                   dialog.reset_button, dialog.cancel_button):
        assert widget.isVisible()
        assert dialog.rect().contains(widget.geometry())
        assert widget.height() >= widget.minimumSizeHint().height()
    assert dialog.sample_rate_combobox.isEditable()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_barcode_keyboard_scan_keeps_original_and_existing_enter_policy(dialog, qapp, key):
    dialog.show()
    dialog.activateWindow()
    dialog.barcode_lineedit.setFocus()
    qapp.processEvents()
    QTest.keyClicks(dialog.barcode_lineedit, "ABC/123%2FXY")
    QTest.keyClick(dialog.barcode_lineedit, key)
    qapp.processEvents()
    assert dialog.barcode_lineedit.text() == "ABC/123%2FXY"
    assert dialog.result() != dialog.Accepted
    QTest.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.result() == dialog.Accepted
    assert dialog.filter_config == {"select_barcode": "ABC/123%2FXY"}


@pytest.mark.parametrize("text,expected", [
    ("4000", [4000]),
    ("8000, 102400", [8000, 102400]),
    ("192000", [192000]),
    ("96000", [96000]),
    ("44100, 48000", [44100, 48000]),
    (" 96000，192000,96000 ", [96000, 192000]),
])
def test_custom_sample_rates_apply_and_reopen(dialog, rows, text, expected):
    dialog.sample_rate_combobox.setEditText(text)
    dialog.apply_filters()
    assert dialog.result() == dialog.Accepted
    assert dialog.filter_config == {"select_sample_rate": expected}
    reopened = ArchiveAudioFilterDialog(rows, dialog.metadata_by_id, dialog.filter_config)
    reopened.apply_filters()
    assert reopened.filter_config == dialog.filter_config
    reopened.close()


@pytest.mark.parametrize("text", [
    "", "0", "-1", "3999", "192001", "96000,192001",
    "44100.5", "48kHz", "44100,", "ALL,48000",
])
def test_invalid_sample_rates_keep_dialog_open_and_filters_unchanged(dialog, monkeypatch, text):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog.sample_rate_combobox.setEditText(text)
    dialog.apply_filters()
    assert dialog.result() != dialog.Accepted
    assert dialog.filter_config == {}
    assert len(warnings) == 1
    assert "正整数采样率" in warnings[0]
    assert "4000–192000" in warnings[0]


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_sample_rate_editor_first_enter_keeps_input_then_second_applies(dialog, qapp, key):
    dialog.show()
    dialog.activateWindow()
    editor = dialog.sample_rate_combobox.lineEdit()
    editor.setFocus()
    qapp.processEvents()
    editor.selectAll()
    QTest.keyClicks(editor, "96000")
    QTest.keyClick(editor, key)
    qapp.processEvents()
    assert dialog.result() != dialog.Accepted
    assert editor.text() == "96000"
    QTest.keyClick(editor, key)
    qapp.processEvents()
    assert dialog.result() == dialog.Accepted
    assert dialog.filter_config == {"select_sample_rate": [96000]}


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


def test_real_archive_project_filter_uses_full_directory_identity(archive):
    dialog = archive.create_filter_dialog({})
    project = dialog.metadata_by_id["one"].project_key
    choose(dialog.combos["select_project"], project)
    dialog.apply_filters()
    archive.filter_audio_data_at_filter_config(dialog.filter_config)
    assert [row[0] for row in archive.filter_audio_data] == ["one", "unlabeled"]
    dialog.close()


def test_archive_entry_applies_real_dialog_and_resets_equal_count_filter(archive, monkeypatch):
    assert isinstance(archive.create_filter_dialog({}), ArchiveAudioFilterDialog)
    def execute(dialog):
        choose(dialog.combos["select_station"], "工位1")
        choose(dialog.combos["select_condition"], "正转")
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


def test_custom_sample_rate_filters_real_archive_records(archive):
    archive.all_audio_data.extend([
        audio_row("custom", current_recording_path(), rate=96000),
        audio_row("higher", current_recording_path(), rate=192000),
        audio_row("below_range", current_recording_path(), rate=3999),
        audio_row("above_range", current_recording_path(), rate=192001),
    ])
    dialog = archive.create_filter_dialog({})
    combo = dialog.sample_rate_combobox
    assert "96000" in [combo.itemText(i) for i in range(combo.count())]
    assert "3999" not in [combo.itemText(i) for i in range(combo.count())]
    assert "192001" not in [combo.itemText(i) for i in range(combo.count())]
    combo.setEditText("96000")
    dialog.apply_filters()
    archive.filter_audio_data_at_filter_config(dialog.filter_config)
    assert [row[0] for row in archive.filter_audio_data] == ["custom"]
    combo.setEditText(" all ")
    dialog.apply_filters()
    archive.filter_audio_data_at_filter_config(dialog.filter_config)
    assert archive.filter_audio_data == archive.all_audio_data
    dialog.close()


def test_barcode_filters_archive_rows_and_show_all_clears_it(archive, monkeypatch):
    archive.all_audio_data.extend([
        audio_row("barcode_one", current_recording_path(), barcode="ABC/123/XY"),
        audio_row("barcode_two", current_recording_path(suffix="_3"), barcode="ABC/123/XY"),
        audio_row("partial", current_recording_path(), barcode="ABC/123/XY-extra"),
    ])

    def forbidden(*args, **kwargs):
        pytest.fail("Barcode filtering must use existing records without new I/O")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)

    def execute(dialog):
        dialog.barcode_lineedit.setText("ABC/123/XY")
        dialog.apply_filters()
        return 1, dialog.filter_config

    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", execute)
    original_rows = list(archive.all_audio_data)
    archive.on_click_filter_btn()
    assert [row[0] for row in archive.filter_audio_data] == ["barcode_one", "barcode_two"]
    assert archive.model().rowCount() == 2
    assert archive.model().item(0, 3).text() == "ABC/123/XY"
    assert archive.model().index(0, archive._analysis_btn_col).data(Qt.UserRole) == (
        original_rows[-3][0], original_rows[-3][1],
    )
    archive.show_all_wave()
    assert archive.filter_config == {}
    assert archive.model().rowCount() == len(original_rows)
    assert archive.all_audio_data == original_rows
    reopened = archive.create_filter_dialog(archive.filter_config)
    assert reopened.barcode_lineedit.text() == ""
    reopened.close()


def test_metadata_is_cached_by_identity_and_refreshed_when_path_changes(archive, monkeypatch):
    import ui.archive_audio_data_dialog as module
    parser = Mock(wraps=parse_archive_audio_filter_metadata)
    monkeypatch.setattr(module, "parse_archive_audio_filter_metadata", parser)
    archive.on_clicked_order_btn()
    archive.filter_audio_data_at_filter_config({"select_station": "工位1"})
    assert parser.call_count == 0
    row = archive.all_audio_data[0]
    archive.all_audio_data[0] = (row[0], current_recording_path(station="工位3"), *row[2:])
    archive.filter_audio_data_at_filter_config({"select_station": "工位3"})
    assert parser.call_count == 1
    assert [item[0] for item in archive.filter_audio_data] == [row[0]]
    archive.delete_audio_data_with_id([row[0]])
    assert row[0] not in archive._audio_filter_cache


def test_filtering_does_not_open_database_or_scan_for_new_files(archive, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Applying filters must use the existing in-memory records")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    original = list(archive.all_audio_data)
    dialog = archive.create_filter_dialog({})
    choose(dialog.combos["select_station"], "工位1")
    choose(dialog.combos["select_condition"], "正转")
    dialog.apply_filters()
    archive.filter_audio_data_at_filter_config(dialog.filter_config)
    assert archive.all_audio_data == original
    assert len(archive.filter_audio_data) == 2
    dialog.close()


def test_no_match_is_empty_and_cancel_preserves_selection(archive, monkeypatch):
    archive.model().item(0, 0).setCheckState(Qt.Checked)
    selected = dict(archive.select_wave_data)
    def cancel_draft(dialog):
        choose(dialog.combos["select_station"], "工位2")
        dialog.reject()
        return 0, {}
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", cancel_draft)
    archive.on_click_filter_btn()
    assert archive.select_wave_data == selected
    assert archive.model().rowCount() == 4
    def apply_unmatched_combination(dialog):
        choose(dialog.combos["select_station"], "工位1")
        choose(dialog.combos["select_condition"], "反转")
        dialog.apply_filters()
        return 1, dialog.filter_config
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", apply_unmatched_combination)
    archive.on_click_filter_btn()
    assert archive.filter_audio_data == []
    assert archive.model().rowCount() == 0
    assert archive.select_wave_data == {}


def test_legacy_filter_matches_labels_in_label_column_only():
    state = Mock()
    state.all_audio_data = [audio_row("OK", label="not_labeled"), audio_row("real_ok", label="OK")]
    AudioDataManageDialog.filter_audio_data_at_filter_config(state, {"select_labels": ["OK"]})
    assert [row[0] for row in state.filter_audio_data] == ["real_ok"]
