import os
import shutil
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QApplication, QHeaderView, QSizePolicy, QTableView, QTableWidgetItem

from consts import error_code


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeConfigManager:
    def __init__(self, config):
        self.config = config
        self.saved = []

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config_data):
        self.saved.append((model_type, config_data))
        return True


def stub_pattern_match_import_dependencies(monkeypatch):
    class FakeLoadUiConfig:
        @staticmethod
        def load_data_from_json(_file_path):
            return 0, {}

    class FakeFileOps:
        @staticmethod
        def get_relative_path(file_path, _base_path):
            return file_path

    class FakeDataView(QTableView):
        def __init__(self, rows, columns, _data):
            super().__init__()
            self._model = QStandardItemModel(rows, columns)
            self.setModel(self._model)

        def set_h_header(self, labels):
            self._model.setHorizontalHeaderLabels(labels)

    class FakeAudioClipExtractionDialog:
        def __init__(self, *args, **kwargs):
            pass

        def on_exec(self):
            return None, None, None

    class FakeGenericFeatureParamsDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return 0

        def get_params(self):
            return {}

    monkeypatch.setitem(sys.modules, "base.load_config", types.SimpleNamespace(LoadUiConfig=FakeLoadUiConfig))
    monkeypatch.setitem(sys.modules, "base.file_ops", types.SimpleNamespace(FileOps=FakeFileOps))
    monkeypatch.setitem(sys.modules, "base.load_audio", types.SimpleNamespace(load_audio_simple=lambda *_: ([], None)))
    monkeypatch.setitem(
        sys.modules,
        "ui.custom_ui_widget.custom_table_widget",
        types.SimpleNamespace(DataView=FakeDataView),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.custom_ui_widget.audio_clip_extraction_dialog",
        types.SimpleNamespace(AudioClipExtractionDialog=FakeAudioClipExtractionDialog),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.generic_feature_params_dialog",
        types.SimpleNamespace(GenericFeatureParamsDialog=FakeGenericFeatureParamsDialog),
    )


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def local_tmp_path(request):
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in request.node.name)
    path = REPO_ROOT / ".tmp" / "analysis_config_dialog_migration" / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        for parent in (path.parent, path.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                pass


def assert_vertical_golden_size(window):
    assert window.minimumWidth() == 630
    assert window.minimumHeight() == 840
    assert round(window.minimumWidth() / window.minimumHeight(), 2) == 0.75


def create_threshold_widget(qapp, load_config):
    from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget

    widget = ThresholdConfigWidget(
        load_config=load_config,
        model_type="SPL",
        allow_manual_limits=True,
    )
    widget.show()
    qapp.processEvents()
    return widget


def create_manual_dialog(widget):
    dialog = widget._create_manual_limit_dialog()
    dialog.show()
    QApplication.instance().processEvents()
    return dialog


def confirm_dialog(dialog):
    dialog._on_confirm_clicked()
    QApplication.instance().processEvents()


def plot_data_snapshot(plot_widget):
    snapshot = []
    for item in plot_widget.listDataItems():
        x_data, y_data = item.getData()
        snapshot.append((np.asarray(x_data).tolist(), np.asarray(y_data).tolist()))
    return snapshot


def _manual_table_content_height(table):
    height = (
        table.horizontalHeader().height()
        + sum(table.rowHeight(row) for row in range(table.rowCount()))
        + table.frameWidth() * 2
    )
    scroll_bar = table.horizontalScrollBar()
    if scroll_bar.maximum() > scroll_bar.minimum():
        height += scroll_bar.sizeHint().height()
    return height


def set_table_cell(table, row, column, value):
    item = table.item(row, column)
    if item is None:
        item = QTableWidgetItem()
        table.setItem(row, column, item)
    item.setText(str(value))
    QApplication.instance().processEvents()


def set_segment_column(table, column, values):
    for row, value in enumerate(values):
        set_table_cell(table, row, column, value)


def clear_segment_column(table, column):
    for row in range(table.rowCount()):
        set_table_cell(table, row, column, "")


def assert_blank_column(table, column):
    assert [table.item(row, column).text() if table.item(row, column) else "" for row in range(4)] == ["", "", "", ""]


def manual_table_column_texts(table, column):
    return [table.item(row, column).text() if table.item(row, column) else "" for row in range(4)]


def manual_table_snapshot(table):
    return [manual_table_column_texts(table, column) for column in range(table.columnCount())]


def select_columns(table, columns):
    if columns:
        table.setCurrentCell(0, columns[0])
    selection = QItemSelection()
    for column in columns:
        top = table.model().index(0, column)
        bottom = table.model().index(table.rowCount() - 1, column)
        selection.select(top, bottom)
    table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)
    QApplication.instance().processEvents()


def assert_manual_table_shape_refreshed(table):
    assert [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())] == [
        str(column + 1) for column in range(table.columnCount())
    ]
    if table.columnCount() <= 7:
        assert_manual_table_uses_stretch(table, expected_columns=table.columnCount())
    else:
        assert_manual_table_uses_interactive_80px_columns(table, expected_columns=table.columnCount())
    assert table.height() == _manual_table_content_height(table)


def test_manual_segment_tables_use_single_selection(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert table.selectionMode() == table.SingleSelection
        assert table.selectionBehavior() == table.SelectItems


def test_threshold_widget_manual_mode_without_segment_keys_starts_blank_not_scalar_derived(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_upper": 88.5,
            "manual_lower_enabled": True,
            "manual_lower": 20.0,
        },
    )

    assert widget.manual_edit_button.isVisible() is True
    assert not hasattr(widget, "manual_upper_table")
    assert not hasattr(widget, "manual_lower_table")
    dialog = create_manual_dialog(widget)

    assert dialog.editor.manual_upper_table.rowCount() == 4
    assert dialog.editor.manual_lower_table.rowCount() == 4
    assert dialog.editor.manual_upper_table.columnCount() == 1
    assert dialog.editor.manual_lower_table.columnCount() == 1
    assert_blank_column(dialog.editor.manual_upper_table, 0)
    assert_blank_column(dialog.editor.manual_lower_table, 0)
    assert not hasattr(widget, "manual_upper_spin")
    assert not hasattr(widget, "manual_lower_spin")


def test_threshold_widget_filling_final_upper_column_auto_appends_blank_column(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)

    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])

    assert dialog.editor.manual_upper_table.columnCount() == 2
    assert_blank_column(dialog.editor.manual_upper_table, 1)


def test_threshold_widget_partial_non_trailing_column_is_invalid_and_warns(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    set_table_cell(dialog.editor.manual_upper_table, 0, 0, "")

    confirm_dialog(dialog)
    assert warnings
    assert dialog.result() != dialog.Accepted


def test_threshold_widget_blank_before_later_nonblank_column_is_invalid_and_warns(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    set_segment_column(dialog.editor.manual_upper_table, 1, [2, 30, 3, 40])
    clear_segment_column(dialog.editor.manual_upper_table, 0)

    confirm_dialog(dialog)
    assert warnings
    assert dialog.result() != dialog.Accepted


def test_threshold_widget_manual_config_serializes_segments_without_scalar_keys(qapp, monkeypatch):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    monkeypatch.setattr(dialog, "exec_", lambda: (dialog._on_confirm_clicked() or dialog.result()))
    monkeypatch.setattr(widget, "_create_manual_limit_dialog", lambda: dialog)
    widget.manual_edit_button.click()
    qapp.processEvents()
    config = widget.get_config()

    assert config["limit_mode"] == "manual"
    assert config["manual_upper_enabled"] is True
    assert config["manual_lower_enabled"] is False
    assert config["manual_upper_segments"] == [{"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0}]
    assert config["manual_lower_segments"] == []
    assert "manual_upper" not in config
    assert "manual_lower" not in config


def test_threshold_widget_threshold_unchecked_hides_limit_group(qapp):
    widget = create_threshold_widget(qapp, {"limit_checked": False, "limit_mode": "manual"})

    assert widget.limit_group_box.isVisible() is False


def test_threshold_widget_mode_visibility_switches_between_csv_and_manual(qapp):
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual"})

    assert widget.manual_widget.isVisible() is True
    assert widget.manual_edit_button.isVisible() is True
    assert widget.config_dir_box.isVisible() is False

    widget.csv_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.manual_widget.isVisible() is False
    assert widget.manual_edit_button.isVisible() is False
    assert widget.config_dir_box.isVisible() is True


def test_threshold_widget_unchecked_upper_lower_hide_their_tables(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
        },
    )
    dialog = create_manual_dialog(widget)

    assert dialog.editor.manual_upper_table.isVisible() is True
    assert dialog.editor.manual_lower_table.isVisible() is True

    dialog.editor.manual_upper_check.setChecked(False)
    dialog.editor.manual_lower_check.setChecked(False)
    qapp.processEvents()

    assert dialog.editor.manual_upper_table.isVisible() is False
    assert dialog.editor.manual_lower_table.isVisible() is False


def test_manual_segment_tables_fit_rendered_content_without_bottom_blank(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_segments": [],
        },
    )
    dialog = create_manual_dialog(widget)

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert table.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
        assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert table.height() == _manual_table_content_height(table)


def manual_segments(count):
    return [
        {"start_x": index * 100, "start_y": 10 + index, "end_x": index * 100 + 50, "end_y": 20 + index}
        for index in range(count)
    ]


def assert_manual_table_uses_stretch(table, expected_columns):
    header = table.horizontalHeader()
    assert table.columnCount() == expected_columns
    assert header.sectionResizeMode(0) == QHeaderView.Stretch
    assert header.minimumSectionSize() < 80


def assert_manual_table_uses_interactive_80px_columns(table, expected_columns):
    header = table.horizontalHeader()
    assert table.columnCount() == expected_columns
    assert header.sectionResizeMode(0) == QHeaderView.Interactive
    assert header.minimumSectionSize() == 80
    assert header.defaultSectionSize() == 80
    for column in range(table.columnCount()):
        assert table.columnWidth(column) >= 80


def test_manual_segment_tables_use_stretch_until_seven_actual_columns(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_segments": [],
        },
    )
    dialog = create_manual_dialog(widget)

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert_manual_table_uses_stretch(table, expected_columns=1)

    dialog.editor.load_manual_config(
        {
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": manual_segments(7),
            "manual_lower_segments": manual_segments(7),
        }
    )
    qapp.processEvents()

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert_manual_table_uses_interactive_80px_columns(table, expected_columns=8)

    dialog.editor.load_manual_config(
        {
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": manual_segments(6),
            "manual_lower_segments": manual_segments(6),
        }
    )
    qapp.processEvents()

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert_manual_table_uses_stretch(table, expected_columns=7)


def test_manual_segment_tables_use_interactive_80px_columns_after_seven_actual_columns(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": manual_segments(7),
            "manual_lower_segments": manual_segments(7),
        },
    )
    dialog = create_manual_dialog(widget)

    for table in (dialog.editor.manual_upper_table, dialog.editor.manual_lower_table):
        assert_manual_table_uses_interactive_80px_columns(table, expected_columns=8)


def test_manual_segment_table_height_includes_required_horizontal_scrollbar(qapp):
    complete_segments = manual_segments(9)
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": complete_segments,
            "manual_lower_segments": [],
        },
    )
    dialog = create_manual_dialog(widget)
    dialog.setFixedWidth(dialog.minimumSizeHint().width())
    qapp.processEvents()

    table = dialog.editor.manual_upper_table
    scroll_bar = table.horizontalScrollBar()
    assert table.columnCount() == 10
    assert scroll_bar.maximum() > scroll_bar.minimum()
    assert table.height() == _manual_table_content_height(table)


def test_threshold_widget_manual_mode_recomputes_scroll_height_after_being_enabled(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPLF": {
                "splf_calc_mode": "fundamental",
                "octave_smoothing": 0,
                "golden_sample_checked": False,
                "limit_checked": False,
                "limit_mode": "csv",
                "limit_data": None,
                "manual_upper_enabled": True,
                "manual_upper_segments": [],
                "manual_lower_enabled": True,
                "manual_lower_segments": [],
                "weighting": "Z",
                "analysis_channel": 0,
            }
        }
    )
    window = SplConfigWindow(config_manager, "SPLF", available_channels=[0])
    window.show()
    qapp.processEvents()

    widget = window.threshold_widget
    widget.limit_checkbox.setChecked(True)
    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert window.section_container.minimumHeight() >= window.section_container.sizeHint().height()
    assert widget.limit_group_box.height() >= widget.limit_group_box.sizeHint().height()
    assert widget.manual_edit_button.isVisible() is True

    window.close()


def test_threshold_widget_manual_preview_inserts_gap_between_disconnected_segments(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    set_segment_column(dialog.editor.manual_upper_table, 1, [2, 30, 3, 40])

    x_values, upper_values, lower_values = dialog.editor.manual_limit_preview_data()

    assert x_values[:2] == [0.0, 1.0]
    assert np.isnan(x_values[2])
    assert x_values[3:] == [2.0, 3.0]
    assert upper_values[:2] == [10.0, 20.0]
    assert np.isnan(upper_values[2])
    assert upper_values[3:] == [30.0, 40.0]
    assert np.all(np.isnan(lower_values))


def test_manual_limit_preview_highlight_current_upper_column_only(qapp):
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [1, 20, 2, 30])
    set_segment_column(table, 2, [2, 30, 3, 40])
    table.setSelectionMode(table.MultiSelection)
    select_columns(table, [0, 2])
    table.setSelectionMode(table.SingleSelection)
    table.setCurrentCell(0, 2)
    QApplication.instance().processEvents()

    dialog._refresh_preview()

    items = plot_data_snapshot(dialog.limit_graph)
    np.testing.assert_equal(
        items[0],
        ([0.0, 1.0, np.nan, 1.0, 2.0, np.nan, 2.0, 3.0], [10.0, 20.0, np.nan, 20.0, 30.0, np.nan, 30.0, 40.0]),
    )
    assert items[1:] == [([2.0, 3.0], [30.0, 40.0])]


def test_manual_limit_preview_highlight_current_lower_column_only_when_lower_active(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    upper = dialog.editor.manual_upper_table
    lower = dialog.editor.manual_lower_table
    set_segment_column(upper, 0, [0, 10, 1, 20])
    set_segment_column(lower, 0, [0, 1, 1, 2])
    upper.setCurrentCell(0, 0)
    lower.setCurrentCell(0, 0)
    lower.setFocus()
    QApplication.instance().processEvents()

    dialog._refresh_preview()

    highlight_items = plot_data_snapshot(dialog.limit_graph)[2:]
    assert highlight_items == [([0.0, 1.0], [1.0, 2.0])]


def test_manual_limit_preview_highlight_focus_switch_refreshes_current_column(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    upper = dialog.editor.manual_upper_table
    lower = dialog.editor.manual_lower_table
    set_segment_column(upper, 0, [0, 10, 1, 20])
    set_segment_column(lower, 0, [0, 1, 1, 2])
    upper.setCurrentCell(0, 0)
    lower.setCurrentCell(0, 0)

    upper.setFocus()
    QApplication.instance().processEvents()
    assert plot_data_snapshot(dialog.limit_graph)[2:] == [([0.0, 1.0], [10.0, 20.0])]

    lower.setFocus()
    QApplication.instance().processEvents()
    assert plot_data_snapshot(dialog.limit_graph)[2:] == [([0.0, 1.0], [1.0, 2.0])]


def test_manual_limit_preview_highlight_omits_blank_partial_invalid_and_disabled_current_columns(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": False},
    )
    dialog = create_manual_dialog(widget)
    upper = dialog.editor.manual_upper_table
    lower = dialog.editor.manual_lower_table
    set_segment_column(upper, 0, [0, 10, 1, 20])
    upper.setColumnCount(4)
    dialog.editor._ensure_manual_table_items(upper)
    dialog.editor._update_manual_table_headers(upper)
    set_table_cell(upper, 0, 1, "2")
    set_segment_column(upper, 2, [3, "bad", 4, 40])
    set_segment_column(lower, 0, [0, 1, 1, 2])

    for column in (1, 2, 3):
        upper.setCurrentCell(0, column)
        QApplication.instance().processEvents()
        dialog._refresh_preview()
        assert plot_data_snapshot(dialog.limit_graph)[1:] == []

    upper.setCurrentCell(0, 0)
    QApplication.instance().processEvents()
    dialog._refresh_preview()
    assert plot_data_snapshot(dialog.limit_graph)[1:] == [([0.0, 1.0], [10.0, 20.0])]

    lower.setCurrentCell(0, 0)
    QApplication.instance().processEvents()
    dialog._refresh_preview()

    assert len(plot_data_snapshot(dialog.limit_graph)) == 1


def test_manual_limit_preview_highlight_uses_current_column(qapp):
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [1, 20, 2, 30])
    table.clearSelection()
    table.setCurrentCell(0, 1)
    QApplication.instance().processEvents()

    dialog._refresh_preview()

    assert plot_data_snapshot(dialog.limit_graph)[1:] == [([1.0, 2.0], [20.0, 30.0])]


def test_manual_limit_preview_highlight_updates_after_current_cell_edit(qapp):
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    table.setCurrentCell(1, 0)
    QApplication.instance().processEvents()
    dialog._refresh_preview()
    assert plot_data_snapshot(dialog.limit_graph)[1:] == [([0.0, 1.0], [10.0, 20.0])]

    set_table_cell(table, 1, 0, "15")
    dialog._refresh_preview()

    assert plot_data_snapshot(dialog.limit_graph)[1:] == [([0.0, 1.0], [15.0, 20.0])]


def test_manual_limit_preview_highlight_ignores_out_of_range_current_column(qapp, monkeypatch):
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    table.setCurrentCell(0, 0)
    monkeypatch.setattr(table, "currentColumn", lambda: 99)
    dialog.editor._remember_active_manual_table(table)
    QApplication.instance().processEvents()

    dialog._refresh_preview()

    assert plot_data_snapshot(dialog.limit_graph) == [([0.0, 1.0], [10.0, 20.0])]


def test_threshold_widget_switching_to_csv_without_data_clears_manual_preview(qapp, monkeypatch):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "limit_data": None,
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    monkeypatch.setattr(dialog, "exec_", lambda: (dialog._on_confirm_clicked() or dialog.result()))
    monkeypatch.setattr(widget, "_create_manual_limit_dialog", lambda: dialog)
    widget.manual_edit_button.click()
    qapp.processEvents()

    assert len(widget.limit_graph.listDataItems()) == 1

    widget.csv_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.current_limit_mode() == "csv"
    assert widget.limit_data is None
    assert widget.limit_graph.listDataItems() == []


def test_threshold_widget_switch_to_manual_seeds_from_loaded_limit_data_when_no_saved_segments(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [np.nan, np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    config = widget.get_config()
    assert config["limit_mode"] == "manual"
    assert config["manual_upper_enabled"] is True
    assert config["manual_lower_enabled"] is False
    assert config["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
        {"start_x": 1.0, "start_y": 20.0, "end_x": 2.0, "end_y": 30.0},
    ]
    assert config["manual_lower_segments"] == []


def test_threshold_widget_manual_toggle_emits_seeded_config_to_sync_listener(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [np.nan, np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    emitted_configs = []
    widget.config_changed.connect(lambda: emitted_configs.append(widget.get_config()))

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    expected_segments = [
        {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
        {"start_x": 1.0, "start_y": 20.0, "end_x": 2.0, "end_y": 30.0},
    ]
    manual_configs = [config for config in emitted_configs if config["limit_mode"] == "manual"]
    assert len(manual_configs) == 1
    assert manual_configs[0]["manual_upper_segments"] == expected_segments


def test_threshold_widget_loaded_limit_data_does_not_overwrite_saved_manual_segments(qapp):
    saved = [{"start_x": 0.0, "start_y": 99.0, "end_x": 1.0, "end_y": 99.0}]
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0], [10.0, 20.0], [np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": saved,
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.get_config()["manual_upper_segments"] == saved


def test_threshold_widget_failed_csv_seed_warns_and_keeps_accepted_manual_config(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    saved = [{"start_x": 0.0, "start_y": 99.0, "end_x": 1.0, "end_y": 99.0}]
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0, 1.0], [10.0, 20.0, 30.0], [np.nan, np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": saved,
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    widget._pending_manual_seed_data = widget.limit_data

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert warnings
    assert "CSV阈值无法自动转换为手动分段" in warnings[-1][2]
    assert widget.get_config()["manual_upper_segments"] == saved


def test_threshold_widget_failed_csv_seed_warns_once_for_manual_toggle(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    saved = [{"start_x": 0.0, "start_y": 99.0, "end_x": 1.0, "end_y": 99.0}]
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0, 1.0], [10.0, 20.0, 30.0], [np.nan, np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": saved,
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    widget._pending_manual_seed_data = widget.limit_data

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert len(warnings) == 1
    assert "CSV阈值无法自动转换为手动分段" in warnings[0][2]


def test_threshold_widget_newly_imported_csv_sets_pending_manual_seed(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    path = local_tmp_path / "limit.csv"
    path.write_text("x,upperbound\n0,10\n1,20\n2,30\n", encoding="utf-8")
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: None)
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), "CSV 文件 (*.csv)"),
    )

    widget._on_config_dir_btn_clicked()
    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.get_config()["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
        {"start_x": 1.0, "start_y": 20.0, "end_x": 2.0, "end_y": 30.0},
    ]


def test_threshold_widget_lower_only_csv_seed_disables_upper_and_enables_lower(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    path = local_tmp_path / "limit.csv"
    path.write_text("x,lowerbound\n0,1\n1,2\n2,3\n", encoding="utf-8")
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: None)
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), "CSV 文件 (*.csv)"),
    )

    widget._on_config_dir_btn_clicked()
    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    config = widget.get_config()
    assert config["manual_upper_enabled"] is False
    assert config["manual_upper_segments"] == []
    assert config["manual_lower_enabled"] is True
    assert config["manual_lower_segments"] == [
        {"start_x": 0.0, "start_y": 1.0, "end_x": 1.0, "end_y": 2.0},
        {"start_x": 1.0, "start_y": 2.0, "end_x": 2.0, "end_y": 3.0},
    ]


def test_threshold_widget_two_sided_csv_seed_handles_blank_cells_per_side(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    path = local_tmp_path / "limit.csv"
    path.write_text(
        "x,upperbound,lowerbound\n0,10,\n1,20,5\n2,30,6\n3,,7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: None)
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "csv",
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), "CSV 文件 (*.csv)"),
    )

    widget._on_config_dir_btn_clicked()
    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    config = widget.get_config()
    assert config["manual_upper_enabled"] is True
    assert config["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
        {"start_x": 1.0, "start_y": 20.0, "end_x": 2.0, "end_y": 30.0},
    ]
    assert config["manual_lower_enabled"] is True
    assert config["manual_lower_segments"] == [
        {"start_x": 1.0, "start_y": 5.0, "end_x": 2.0, "end_y": 6.0},
        {"start_x": 2.0, "start_y": 6.0, "end_x": 3.0, "end_y": 7.0},
    ]


def test_threshold_widget_initial_manual_mode_seeds_loaded_limit_data_and_preview(qapp):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "limit_data": ([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [np.nan, np.nan, np.nan]),
            "manual_upper_enabled": True,
            "manual_upper_segments": [],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
        },
    )

    config = widget.get_config()
    assert config["limit_mode"] == "manual"
    assert config["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
        {"start_x": 1.0, "start_y": 20.0, "end_x": 2.0, "end_y": 30.0},
    ]
    assert len(widget.limit_graph.listDataItems()) == 1
    x_data, y_data = widget.limit_graph.listDataItems()[0].getData()
    x_values = np.asarray(x_data).tolist()
    y_values = np.asarray(y_data).tolist()
    assert x_values[:2] == [0.0, 1.0]
    assert np.isnan(x_values[2])
    assert x_values[3:] == [1.0, 2.0]
    assert y_values[:2] == [10.0, 20.0]
    assert np.isnan(y_values[2])
    assert y_values[3:] == [20.0, 30.0]


def test_manual_limit_dialog_reject_keeps_parent_config_unchanged(qapp, monkeypatch):
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": [{"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0}],
            "manual_lower_segments": [],
        },
    )
    before = widget.get_config()
    before_plot = plot_data_snapshot(widget.limit_graph)

    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 99, 1, 100])
    monkeypatch.setattr(dialog, "exec_", lambda: dialog.Rejected)
    monkeypatch.setattr(widget, "_create_manual_limit_dialog", lambda: dialog)
    widget.manual_edit_button.click()
    qapp.processEvents()

    assert widget.get_config() == before
    assert plot_data_snapshot(widget.limit_graph) == before_plot


def test_manual_limit_dialog_button_confirm_updates_parent_config_and_preview(qapp, monkeypatch):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )

    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])

    def accept_from_button_path():
        dialog._on_confirm_clicked()
        return dialog.result()

    monkeypatch.setattr(dialog, "exec_", accept_from_button_path)
    monkeypatch.setattr(widget, "_create_manual_limit_dialog", lambda: dialog)
    widget.manual_edit_button.click()
    qapp.processEvents()

    config = widget.get_config()
    assert config["manual_upper_segments"] == [{"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0}]
    assert len(widget.limit_graph.listDataItems()) == 1


def test_manual_limit_dialog_confirm_replaces_accepted_segments_without_stale_preview(qapp, monkeypatch):
    stale_segment = {"start_x": 2.0, "start_y": 30.0, "end_x": 3.0, "end_y": 40.0}
    replacement_segments = [{"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0}]
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": [replacement_segments[0], stale_segment],
            "manual_lower_segments": [],
        },
    )

    dialog = create_manual_dialog(widget)
    clear_segment_column(dialog.editor.manual_upper_table, 1)

    def accept_from_button_path():
        dialog._on_confirm_clicked()
        return dialog.result()

    monkeypatch.setattr(dialog, "exec_", accept_from_button_path)
    monkeypatch.setattr(widget, "_create_manual_limit_dialog", lambda: dialog)
    widget.manual_edit_button.click()
    qapp.processEvents()

    config = widget.get_config()
    assert config["manual_upper_segments"] == replacement_segments
    assert stale_segment not in config["manual_upper_segments"]
    assert_blank_column(widget._manual_state_editor.manual_upper_table, 1)
    assert plot_data_snapshot(widget.limit_graph) == [([0.0, 1.0], [10.0, 20.0])]


def test_manual_limit_dialog_exports_current_valid_table_to_csv(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    export_path = local_tmp_path / "manual.csv"
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV 文件 (*.csv)"),
    )

    dialog._on_export_clicked()

    assert export_path.read_text(encoding="utf-8").splitlines() == [
        "x,upperbound",
        "0.0,10.0",
        "1.0,20.0",
    ]
    assert dialog.isVisible() is True
    assert dialog.result() != dialog.Accepted
    assert widget.get_config()["manual_upper_segments"] == []


def test_manual_limit_dialog_exports_lower_only_csv(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": False,
            "manual_lower_enabled": True,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_lower_table, 0, [0, 1, 1, 2])
    export_path = local_tmp_path / "manual_lower.csv"
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV 文件 (*.csv)"),
    )

    dialog._on_export_clicked()

    assert export_path.read_text(encoding="utf-8").splitlines() == [
        "x,lowerbound",
        "0.0,1.0",
        "1.0,2.0",
    ]
    assert dialog.isVisible() is True


def test_manual_limit_dialog_exports_two_sided_discontinuity_with_blanks(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    set_segment_column(dialog.editor.manual_upper_table, 1, [1, 30, 2, 40])
    set_segment_column(dialog.editor.manual_lower_table, 0, [1, 5, 2, 6])
    export_path = local_tmp_path / "manual_two_sided.csv"
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV 文件 (*.csv)"),
    )

    dialog._on_export_clicked()

    assert export_path.read_text(encoding="utf-8").splitlines() == [
        "x,upperbound,lowerbound",
        "0.0,10.0,",
        "1.0,20.0,",
        "1.0,30.0,5.0",
        "2.0,40.0,6.0",
    ]
    assert dialog.isVisible() is True


def test_manual_limit_dialog_export_invalid_data_warns_without_creating_file(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: pytest.fail("save dialog should not open for invalid manual data"),
    )
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
        },
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 0, 1, 0])
    set_segment_column(dialog.editor.manual_lower_table, 0, [0, 1, 1, 0])

    dialog._on_export_clicked()

    assert warnings
    assert "下限不能大于上限" in warnings[-1][2]
    assert list(local_tmp_path.iterdir()) == []
    assert dialog.isVisible() is True


def test_manual_limit_dialog_export_cancel_keeps_dialog_open_without_writing(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )

    dialog._on_export_clicked()

    assert list(local_tmp_path.iterdir()) == []
    assert dialog.isVisible() is True
    assert dialog.result() != dialog.Accepted


def test_manual_limit_dialog_export_io_error_warns_and_keeps_dialog_open(qapp, monkeypatch, local_tmp_path):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    export_path = local_tmp_path / "manual.csv"
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(
        threshold_config_widget.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV 文件 (*.csv)"),
    )

    def raise_io_error(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(threshold_config_widget, "write_manual_config_csv", raise_io_error)
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])

    dialog._on_export_clicked()

    assert warnings
    assert "导出CSV失败" in warnings[-1][2]
    assert "disk full" in warnings[-1][2]
    assert dialog.isVisible() is True
    assert dialog.result() != dialog.Accepted


def test_manual_table_clear_table_resets_only_clicked_table(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 10, 1, 20])
    set_segment_column(dialog.editor.manual_lower_table, 0, [0, 1, 1, 2])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._clear_manual_table_data(dialog.editor.manual_upper_table)

    assert emissions == ["changed"]
    assert dialog.editor.manual_upper_table.columnCount() == 1
    assert_blank_column(dialog.editor.manual_upper_table, 0)
    assert dialog.editor.manual_lower_table.columnCount() == 2
    assert manual_table_column_texts(dialog.editor.manual_lower_table, 0) == ["0", "1", "1", "2"]
    assert_manual_table_shape_refreshed(dialog.editor.manual_upper_table)


def test_manual_table_clear_column_clears_only_clicked_column_and_refreshes_shape(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [2, 30, 3, 40])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._clear_manual_table_column(table, 0)

    assert emissions == ["changed"]
    assert_blank_column(table, 0)
    assert manual_table_column_texts(table, 1) == ["2", "30", "3", "40"]
    assert_blank_column(table, 2)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_clear_last_nonblank_column_removes_redundant_blank_tail(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [2, 30, 3, 40])
    table.setColumnCount(4)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._clear_manual_table_column(table, 1)

    assert emissions == ["changed"]
    assert table.columnCount() == 2
    assert manual_table_column_texts(table, 0) == ["0", "10", "1", "20"]
    assert_blank_column(table, 1)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_nonblank_column_removes_column_and_keeps_blank_tail(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [2, 30, 3, 40])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_column(table, 0)

    assert emissions == ["changed"]
    assert table.columnCount() == 2
    assert manual_table_column_texts(table, 0) == ["2", "30", "3", "40"]
    assert_blank_column(table, 1)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_clicked_column_only_even_when_other_columns_selected(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [1, 20, 2, 30])
    set_segment_column(table, 2, [2, 30, 3, 40])
    set_segment_column(table, 3, [3, 40, 4, 50])
    table.setSelectionMode(table.MultiSelection)
    select_columns(table, [1, 3])
    table.setSelectionMode(table.SingleSelection)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_columns(table, clicked_column=1)

    assert emissions == ["changed"]
    assert manual_table_snapshot(table) == [
        ["0", "10", "1", "20"],
        ["2", "30", "3", "40"],
        ["3", "40", "4", "50"],
        ["", "", "", ""],
    ]
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_clicked_column_only_when_clicked_column_is_not_selected(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [1, 20, 2, 30])
    set_segment_column(table, 2, [2, 30, 3, 40])
    table.setSelectionMode(table.MultiSelection)
    select_columns(table, [1, 2])
    table.setSelectionMode(table.SingleSelection)

    dialog.editor._delete_manual_table_columns(table, clicked_column=0)

    assert manual_table_snapshot(table) == [
        ["1", "20", "2", "30"],
        ["2", "30", "3", "40"],
        ["", "", "", ""],
    ]


def test_manual_table_delete_blank_middle_column(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [2, 30, 3, 40])
    dialog.editor._insert_manual_table_column_after(table, 0)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_columns(table, clicked_column=1)

    assert emissions == ["changed"]
    assert manual_table_snapshot(table) == [
        ["0", "10", "1", "20"],
        ["2", "30", "3", "40"],
        ["", "", "", ""],
    ]
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_all_selected_columns_leaves_one_blank_column(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])

    dialog.editor._delete_manual_table_columns(table, clicked_column=0)

    assert table.columnCount() == 1
    assert_blank_column(table, 0)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_only_selected_blank_columns_preserves_one_blank_tail(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    table.setColumnCount(4)
    dialog.editor._ensure_manual_table_items(table)
    dialog.editor._update_manual_table_headers(table)
    table.setSelectionMode(table.MultiSelection)
    select_columns(table, [1, 2, 3])
    table.setSelectionMode(table.SingleSelection)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_columns(table, clicked_column=1)

    assert emissions == ["changed"]
    assert manual_table_snapshot(table) == [
        ["0", "10", "1", "20"],
        ["", "", "", ""],
    ]
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_only_nonblank_column_leaves_one_blank_column(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_column(table, 0)

    assert emissions == ["changed"]
    assert table.columnCount() == 1
    assert_blank_column(table, 0)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_delete_only_nonblank_column_removes_existing_redundant_blank_tail(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    table.setColumnCount(3)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_column(table, 0)

    assert emissions == ["changed"]
    assert table.columnCount() == 1
    assert_blank_column(table, 0)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_single_delete_affects_only_clicked_table(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    upper = dialog.editor.manual_upper_table
    lower = dialog.editor.manual_lower_table
    set_segment_column(upper, 0, [0, 10, 1, 20])
    set_segment_column(upper, 1, [1, 20, 2, 30])
    set_segment_column(lower, 0, [0, 1, 1, 2])
    upper.setSelectionMode(upper.MultiSelection)
    select_columns(upper, [0, 1])
    upper.setSelectionMode(upper.SingleSelection)
    select_columns(lower, [0])

    dialog.editor._delete_manual_table_columns(upper, clicked_column=0)

    assert manual_table_snapshot(upper) == [["1", "20", "2", "30"], ["", "", "", ""]]
    assert manual_table_column_texts(lower, 0) == ["0", "1", "1", "2"]
    assert_blank_column(lower, 1)


def test_manual_table_delete_final_blank_column_recreates_blank_tail(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._delete_manual_table_columns(table, clicked_column=1)

    assert emissions == ["changed"]
    assert manual_table_snapshot(table) == [
        ["0", "10", "1", "20"],
        ["", "", "", ""],
    ]
    assert_manual_table_shape_refreshed(table)


def test_manual_table_insert_blank_column_to_right_and_refreshes_shape(qapp):
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [2, 30, 3, 40])
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._insert_manual_table_column_after(table, 0)

    assert emissions == ["changed"]
    assert table.columnCount() == 4
    assert manual_table_column_texts(table, 0) == ["0", "10", "1", "20"]
    assert_blank_column(table, 1)
    assert manual_table_column_texts(table, 2) == ["2", "30", "3", "40"]
    assert_blank_column(table, 3)
    assert_manual_table_shape_refreshed(table)


def test_manual_table_context_menu_delete_enabled_for_blank_column(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    created_menus = []

    class FakeSignal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class FakeAction:
        def __init__(self, text):
            self._text = text
            self.enabled = True
            self.triggered = FakeSignal()

        def text(self):
            return self._text

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class FakeMenu:
        def __init__(self, _parent=None):
            self.actions = []
            self.exec_pos = None
            created_menus.append(self)

        def addAction(self, text):
            action = FakeAction(text)
            self.actions.append(action)
            return action

        def exec_(self, pos):
            self.exec_pos = pos

    monkeypatch.setattr(threshold_config_widget, "QMenu", FakeMenu, raising=False)
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table

    assert table.contextMenuPolicy() == Qt.CustomContextMenu

    set_segment_column(table, 0, [0, 10, 1, 20])
    dialog.editor._show_manual_table_context_menu(table, table.visualItemRect(table.item(0, 0)).center())
    nonblank_actions = {action.text(): action for action in created_menus[-1].actions}
    assert list(nonblank_actions) == ["清空表格数据", "删除列", "添加列"]
    assert "清空列" not in nonblank_actions
    assert nonblank_actions["删除列"].enabled is True
    assert created_menus[-1].exec_pos is not None

    dialog.editor._show_manual_table_context_menu(table, table.visualItemRect(table.item(0, 1)).center())
    blank_actions = {action.text(): action for action in created_menus[-1].actions}
    assert blank_actions["删除列"].enabled is True


def test_manual_table_context_menu_sets_current_clicked_column(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    created_menus = []

    class FakeSignal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class FakeAction:
        def __init__(self, text):
            self._text = text
            self.enabled = True
            self.triggered = FakeSignal()

        def text(self):
            return self._text

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class FakeMenu:
        def __init__(self, _parent=None):
            self.actions = []
            created_menus.append(self)

        def addAction(self, text):
            action = FakeAction(text)
            self.actions.append(action)
            return action

        def exec_(self, _pos):
            pass

    monkeypatch.setattr(threshold_config_widget, "QMenu", FakeMenu, raising=False)
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    set_segment_column(table, 1, [1, 20, 2, 30])
    table.setCurrentCell(0, 0)
    QApplication.instance().processEvents()

    dialog.editor._show_manual_table_context_menu(table, table.visualItemRect(table.item(2, 1)).center())

    assert table.currentColumn() == 1
    assert dialog.editor.selected_manual_limit_preview_segments() == [([1.0, 2.0], [20.0, 30.0])]


def test_manual_table_context_menu_does_not_emit_when_target_already_active(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    class FakeSignal:
        def connect(self, _callback):
            pass

    class FakeAction:
        def __init__(self, text):
            self._text = text
            self.triggered = FakeSignal()

        def setEnabled(self, _enabled):
            pass

    class FakeMenu:
        def __init__(self, _parent=None):
            pass

        def addAction(self, text):
            return FakeAction(text)

        def exec_(self, _pos):
            pass

    monkeypatch.setattr(threshold_config_widget, "QMenu", FakeMenu, raising=False)
    widget = create_threshold_widget(qapp, {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True})
    dialog = create_manual_dialog(widget)
    table = dialog.editor.manual_upper_table
    set_segment_column(table, 0, [0, 10, 1, 20])
    table.setCurrentCell(0, 0)
    dialog.editor._remember_active_manual_table(table)
    emissions = []
    dialog.editor.config_changed.connect(lambda: emissions.append("changed"))
    emissions.clear()

    dialog.editor._show_manual_table_context_menu(table, table.visualItemRect(table.item(0, 0)).center())

    assert emissions == []


def test_manual_table_context_menu_right_click_other_table_controls_preview_highlight(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    class FakeSignal:
        def connect(self, _callback):
            pass

    class FakeAction:
        def __init__(self, text):
            self._text = text
            self.triggered = FakeSignal()

        def setEnabled(self, _enabled):
            pass

    class FakeMenu:
        def __init__(self, _parent=None):
            pass

        def addAction(self, text):
            return FakeAction(text)

        def exec_(self, _pos):
            pass

    monkeypatch.setattr(threshold_config_widget, "QMenu", FakeMenu, raising=False)
    widget = create_threshold_widget(
        qapp,
        {"limit_checked": True, "limit_mode": "manual", "manual_upper_enabled": True, "manual_lower_enabled": True},
    )
    dialog = create_manual_dialog(widget)
    upper = dialog.editor.manual_upper_table
    lower = dialog.editor.manual_lower_table
    set_segment_column(upper, 0, [0, 10, 1, 20])
    set_segment_column(lower, 0, [0, 1, 1, 2])
    upper.setCurrentCell(0, 0)
    upper.setFocus()
    QApplication.instance().processEvents()

    dialog.editor._show_manual_table_context_menu(lower, lower.visualItemRect(lower.item(0, 0)).center())

    assert lower.hasFocus() or lower.viewport().hasFocus()
    assert dialog.editor.selected_manual_limit_preview_segments() == [([0.0, 1.0], [1.0, 2.0])]


def test_manual_limit_dialog_invalid_warning_is_chinese_and_dialog_stays_open(qapp, monkeypatch):
    from ui.ui_analysis_config import threshold_config_widget

    warnings = []
    monkeypatch.setattr(threshold_config_widget.MessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    widget = create_threshold_widget(
        qapp,
        {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
        },
    )

    dialog = create_manual_dialog(widget)
    set_segment_column(dialog.editor.manual_upper_table, 0, [0, 0, 1, 0])
    set_segment_column(dialog.editor.manual_lower_table, 0, [0, 1, 1, 0])
    confirm_dialog(dialog)

    assert warnings
    assert "下限不能大于上限" in warnings[-1][2]
    assert dialog.result() != dialog.Accepted
    assert dialog.isVisible() is True


def test_threshold_widget_without_manual_limits_keeps_csv_only_behavior(qapp):
    from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget

    widget = ThresholdConfigWidget(
        load_config={"limit_checked": True, "limit_data": None},
        model_type="SPL",
        allow_manual_limits=False,
    )
    widget.show()
    qapp.processEvents()

    assert widget.config_dir_box.isVisible() is True
    assert not hasattr(widget, "csv_mode_radio")
    assert not hasattr(widget, "manual_mode_radio")
    assert not hasattr(widget, "manual_edit_button")
    assert widget.get_config() == {"limit_checked": True, "limit_data": None}


def test_fft_csv_only_threshold_expands_scrollable_section(qapp):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow

    config_manager = FakeConfigManager({"FFT": {"limit_checked": False, "limit_data": None}})
    window = FftConfigWindow(config_manager, "FFT")
    window.show()
    qapp.processEvents()

    widget = window.threshold_widget
    widget.limit_checkbox.setChecked(True)
    qapp.processEvents()

    assert widget.limit_graph.isVisible() is True
    assert widget.maximumHeight() >= widget.sizeHint().height()
    assert window.section_container.minimumHeight() >= window.section_container.sizeHint().height()

    window.close()


def test_ai_dialog_uses_shared_channel_selector_without_changing_config(qapp, monkeypatch):
    class FakeTrainingModelManagement:
        def get_all_model_name_from_db(self):
            return error_code.OK, [("model_a", "1024 samples")]

    monkeypatch.setitem(
        sys.modules,
        "base.training_model_management",
        types.SimpleNamespace(TrainingModelManagement=FakeTrainingModelManagement),
    )
    from ui.ui_analysis_config import ai_config_dialog

    monkeypatch.setattr(ai_config_dialog, "TrainingModelManagement", FakeTrainingModelManagement)
    config_manager = FakeConfigManager({"AI": {"analyse_model_name": "model_a", "analysis_channel": 2}})

    window = ai_config_dialog.AIConfigWindow(config_manager, "AI", available_channels=[0, 2])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute"]
    assert window.get_default_config() == {
        "analyse_model_name": "model_a",
        "analysis_channel": 2,
    }


def test_lp_dialog_preserves_saved_keys_after_channel_migration(qapp):
    from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow

    config_manager = FakeConfigManager(
        {
            "LP": {
                "trigger_threshold": 12,
                "hysterests_threshold": 3,
                "min_check_duration": 20,
                "max_check_duration": 80,
                "loose_particle_num": 2,
                "cutoff_freq": 15000,
                "analysis_channel": 1,
            }
        }
    )

    window = LPConfigWindow(config_manager, "LP", available_channels=[0, 1])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "detection"]
    assert window.get_default_config() == {
        "trigger_threshold": 12,
        "hysterests_threshold": 3,
        "min_check_duration": 20,
        "max_check_duration": 80,
        "loose_particle_num": 2,
        "cutoff_freq": 15000,
        "analysis_channel": 1,
    }


def test_spec_dialog_preserves_saved_keys_after_channel_migration(qapp):
    from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow

    config_manager = FakeConfigManager(
        {
            "Spec": {
                "n_fft": 1024,
                "hop_length": 128,
                "window_func": "hamming",
                "color_map": "magma",
                "freq_scale_type": "log",
                "top_limit": 75,
                "bottom_limit": 35,
                "custom_limit": True,
                "analysis_channel": 3,
            }
        }
    )

    window = SpecConfigWindow(config_manager, "Spec", available_channels=[1, 3])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "display"]
    assert window.get_default_config() == {
        "n_fft": 1024,
        "hop_length": 128,
        "window_func": "hamming",
        "color_map": "magma",
        "freq_scale_type": "log",
        "top_limit": 75,
        "bottom_limit": 35,
        "custom_limit": True,
        "analysis_channel": 3,
    }


def test_spl_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "C",
                "show_overall_spl": True,
                "analysis_channel": 2,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPL", available_channels=[0, 2])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "preprocess", "compute", "judgment"]
    assert window.show_overall_spl_box.text() == "显示总体声压级"
    assert window.show_overall_spl_box.isChecked() is True
    assert window.get_default_config() == {
        "spl_window_unit": "points",
        "spl_window_time_sec": 0.0272,
        "spl_window_points": 1201,
        "smooth_enabled": True,
        "smooth_unit": "points",
        "smooth_time_sec": 0.025,
        "smooth_points": 1102,
        "smooth_algo": 2,
        "smooth_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
        "show_overall_spl": True,
        "weighting": "C",
        "analysis_channel": 2,
        "analysis_time_range_enabled": False,
        "analysis_start_time_sec": 0.0,
        "analysis_end_time_sec": 0.0,
    }


def test_splf_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPLF": {
                "splf_calc_mode": "total",
                "octave_smoothing": 3,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "Z（None）",
                "analysis_channel": 1,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPLF", available_channels=[0, 1])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "reference", "judgment"]
    assert window.show_overall_spl_box is None
    assert window.get_default_config() == {
        "splf_calc_mode": "total",
        "octave_smoothing": 3,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
        "weighting": "Z",
        "analysis_channel": 1,
    }


def test_spl_dialog_preserves_manual_threshold_config(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": False,
                "smooth_enabled": False,
                "limit_checked": True,
                "limit_mode": "manual",
                "limit_data": None,
                "manual_upper_enabled": True,
                "manual_upper_segments": [{"start_x": 0.0, "start_y": 88.5, "end_x": 1.0, "end_y": 88.5}],
                "manual_lower_enabled": True,
                "manual_lower_segments": [{"start_x": 0.0, "start_y": 20.0, "end_x": 1.0, "end_y": 20.0}],
                "weighting": "A",
                "analysis_channel": 0,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPL", available_channels=[0])
    config = window.get_default_config()

    assert config["limit_checked"] is True
    assert config["limit_mode"] == "manual"
    assert config["manual_upper_enabled"] is True
    assert config["manual_upper_segments"] == [{"start_x": 0.0, "start_y": 88.5, "end_x": 1.0, "end_y": 88.5}]
    assert config["manual_lower_enabled"] is True
    assert config["manual_lower_segments"] == [{"start_x": 0.0, "start_y": 20.0, "end_x": 1.0, "end_y": 20.0}]
    assert config["show_overall_spl"] is False
    assert window.threshold_widget.validate() is True


def test_requested_dialogs_return_manual_segment_keys_from_threshold_widget(qapp):
    from ui.ui_analysis_config.fr_config_dialog import FrConfigWindow
    from ui.ui_analysis_config.hd_config_dialog import HdConfigWindow
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
    from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    upper_segments = [{"start_x": 0.0, "start_y": 80.0, "end_x": 1.0, "end_y": 82.0}]
    lower_segments = [{"start_x": 0.0, "start_y": 20.0, "end_x": 1.0, "end_y": 21.0}]

    cases = [
        (
            "SPLF",
            SplConfigWindow(
                FakeConfigManager(
                    {
                        "SPLF": {
                            "splf_calc_mode": "fundamental",
                            "octave_smoothing": 0,
                            "golden_sample_checked": False,
                            "limit_checked": True,
                            "limit_mode": "manual",
                            "limit_data": None,
                            "manual_upper_enabled": True,
                            "manual_upper_segments": upper_segments,
                            "manual_lower_enabled": True,
                            "manual_lower_segments": lower_segments,
                            "weighting": "Z",
                            "analysis_channel": 0,
                        }
                    }
                ),
                "SPLF",
                available_channels=[0],
            ),
        ),
        (
            "FR",
            FrConfigWindow(
                FakeConfigManager(
                    {
                        "FR": {
                            "octave_smoothing": 0,
                            "golden_sample_checked": False,
                            "limit_checked": True,
                            "limit_mode": "manual",
                            "limit_data": None,
                            "manual_upper_enabled": True,
                            "manual_upper_segments": upper_segments,
                            "manual_lower_enabled": True,
                            "manual_lower_segments": lower_segments,
                        }
                    }
                ),
                "FR",
            ),
        ),
        (
            "HD",
            HdConfigWindow(
                FakeConfigManager(
                    {
                        "HD": {
                            "selected_labels": [2],
                            "all_checked": False,
                            "golden_sample_checked": False,
                            "limit_checked": True,
                            "limit_mode": "manual",
                            "limit_data": None,
                            "manual_upper_enabled": True,
                            "manual_upper_segments": upper_segments,
                            "manual_lower_enabled": True,
                            "manual_lower_segments": lower_segments,
                        }
                    }
                ),
                "HD",
            ),
        ),
        (
            "RB",
            RbConfigWindow(
                FakeConfigManager(
                    {
                        "RB": {
                            "selected_labels": [10],
                            "all_checked": False,
                            "golden_sample_checked": False,
                            "limit_checked": True,
                            "limit_mode": "manual",
                            "limit_data": None,
                            "manual_upper_enabled": True,
                            "manual_upper_segments": upper_segments,
                            "manual_lower_enabled": True,
                            "manual_lower_segments": lower_segments,
                        }
                    }
                ),
                "RB",
            ),
        ),
        (
            "PRB",
            PerceptualRbConfigWindow(
                FakeConfigManager(
                    {
                        "PRB": {
                            "masking_config": {"sc_metric": "totalnl_x_ehs"},
                            "golden_sample_checked": False,
                            "limit_checked": True,
                            "limit_mode": "manual",
                            "limit_data": None,
                            "manual_upper_enabled": True,
                            "manual_upper_segments": upper_segments,
                            "manual_lower_enabled": True,
                            "manual_lower_segments": lower_segments,
                        }
                    }
                ),
                "PRB",
            ),
        ),
    ]

    for analysis_type, window in cases:
        assert hasattr(window.threshold_widget, "manual_edit_button"), analysis_type
        assert not hasattr(window.threshold_widget, "manual_upper_table"), analysis_type
        assert not hasattr(window.threshold_widget, "manual_lower_table"), analysis_type
        config = window.get_default_config()
        assert config["limit_mode"] == "manual", analysis_type
        assert config["manual_upper_enabled"] is True, analysis_type
        assert config["manual_upper_segments"] == upper_segments, analysis_type
        assert config["manual_lower_enabled"] is True, analysis_type
        assert config["manual_lower_segments"] == lower_segments, analysis_type
        assert "manual_upper" not in config
        assert "manual_lower" not in config


def test_fr_dialog_uses_shared_octave_smoothing_legacy_fallback(qapp):
    from ui.ui_analysis_config.fr_config_dialog import FrConfigWindow

    config_manager = FakeConfigManager(
        {
            "FR": {
                "smooth_checked": True,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = FrConfigWindow(config_manager, "FR")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["compute", "reference", "judgment"]
    assert window.get_default_config() == {
        "octave_smoothing": 6,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
    }


def test_spl_dialog_restore_default_reloads_semantic_sections(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "A",
                "analysis_channel": 0,
            }
        }
    )
    window = SplConfigWindow(config_manager, "SPL", available_channels=[0, 1])

    config_manager.config = {
        "SPL": {
            "smooth_checked": False,
            "limit_checked": False,
            "limit_data": None,
            "weighting": "C",
            "show_overall_spl": True,
            "analysis_channel": 1,
        }
    }
    window.on_restore_default_btn_clicked()

    assert window.semantic_group_keys() == ["input", "preprocess", "compute", "judgment"]
    assert window.get_default_config() == {
        "spl_window_unit": "points",
        "spl_window_time_sec": 0.0272,
        "spl_window_points": 1201,
        "smooth_enabled": False,
        "smooth_unit": "points",
        "smooth_time_sec": 0.025,
        "smooth_points": 1102,
        "smooth_algo": 2,
        "smooth_checked": False,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
        "show_overall_spl": True,
        "weighting": "C",
        "analysis_channel": 1,
        "analysis_time_range_enabled": False,
        "analysis_start_time_sec": 0.0,
        "analysis_end_time_sec": 0.0,
    }


def test_spl_preprocess_subsections_disable_inactive_controls(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": False,
                "smooth_enabled": False,
                "analysis_time_range_enabled": False,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "Z",
                "analysis_channel": 0,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPL", available_channels=[0])

    assert window.smoothing_section.isChecked() is False
    assert window.time_smoothing_widget.unit_combo.isEnabled() is False
    assert window.time_smoothing_widget.points_spin.isEnabled() is False
    assert window.time_smoothing_widget.algo_group.button(1).isEnabled() is False

    assert window.analysis_time_range_section.isChecked() is False
    assert window.analysis_time_range_widget.start_spin.isEnabled() is False
    assert window.analysis_time_range_widget.end_spin.isEnabled() is False

    window.smoothing_section.setChecked(True)
    window.analysis_time_range_section.setChecked(True)

    assert window.time_smoothing_widget.unit_combo.isEnabled() is True
    assert window.time_smoothing_widget.points_spin.isEnabled() is True
    assert window.time_smoothing_widget.algo_group.button(1).isEnabled() is True
    assert window.analysis_time_range_widget.start_spin.isEnabled() is True
    assert window.analysis_time_range_widget.end_spin.isEnabled() is True
    assert window.get_default_config()["smooth_enabled"] is True
    assert window.get_default_config()["analysis_time_range_enabled"] is True


def test_fba_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow

    config_manager = FakeConfigManager(
        {
            "FBA": {
                "band_strategy": "1/3 倍频程",
                "f_min": 50,
                "f_max": 16000,
                "bandwidth": 200,
                "weighting": "Z（None）",
                "analysis_channel": 3,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = FbaConfigWindow(config_manager, "FBA", available_channels=[1, 3])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "judgment"]
    assert window.get_default_config() == {
        "band_strategy": "1/3 倍频程",
        "f_min": 50,
        "f_max": 16000,
        "bandwidth": 200,
        "analysis_channel": 3,
        "weighting": "Z",
        "limit_checked": False,
        "limit_data": None,
    }


def test_hd_dialog_uses_shared_harmonic_and_golden_widgets(qapp):
    from ui.ui_analysis_config.hd_config_dialog import HdConfigWindow

    config_manager = FakeConfigManager(
        {
            "HD": {
                "selected_labels": [2, "3", 40],
                "all_checked": False,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = HdConfigWindow(config_manager, "HD")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["detection", "reference", "judgment"]
    assert window.get_default_config() == {
        "selected_labels": [2, 3],
        "all_checked": False,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
    }


def test_rb_dialog_filters_harmonics_to_rub_buzz_range(qapp):
    from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow

    config_manager = FakeConfigManager(
        {
            "RB": {
                "selected_labels": [2, 10, "12", 40],
                "all_checked": False,
                "golden_sample_checked": False,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = RbConfigWindow(config_manager, "RB")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["detection", "reference", "judgment"]
    assert window.get_default_config() == {
        "selected_labels": [10, 12],
        "all_checked": False,
        "golden_sample_checked": False,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_upper_enabled": True,
        "manual_upper_segments": [],
        "manual_lower_enabled": False,
        "manual_lower_segments": [],
    }


def test_prb_dialog_preserves_metric_fallback_and_golden_sample(qapp):
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    config_manager = FakeConfigManager(
        {
            "PRB": {
                "masking_config": {"sc_metric": "totalnl_phons", "keep": "value"},
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = PerceptualRbConfigWindow(config_manager, "PRB")
    config = window.get_default_config()

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["compute", "reference", "judgment"]
    assert config["prb_method"] == "sc"
    assert config["masking_config"] == {"sc_metric": "totalnl_x_ehs", "keep": "value"}
    assert config["golden_sample_checked"] is True
    assert config["limit_checked"] is False
    assert config["limit_data"] is None
    assert config["limit_mode"] == "csv"
    assert config["manual_upper_enabled"] is True
    assert config["manual_upper_segments"] == []
    assert config["manual_lower_enabled"] is False
    assert config["manual_lower_segments"] == []


def test_rsc_dialog_keeps_smoothing_key_after_shared_selector_migration(qapp, monkeypatch):
    from ui.ui_analysis_config import reference_spectrum_config_dialog as rsc_dialog

    monkeypatch.setattr(rsc_dialog, "get_reference_data_state", lambda **_: "not_generated")
    config_manager = FakeConfigManager(
        {
            "RSC": {
                "reference_source_path": "",
                "reference_data_path": "",
                "use_custom_band": False,
                "start_freq_hz": 100,
                "end_freq_hz": 1000,
                "highlight_analysis_band": False,
                "enable_threshold_judgment": True,
                "lower_offset_db": -2.0,
                "upper_offset_db": 2.0,
                "channel_labels": {},
                "window": "hann",
                "nperseg": 4096,
                "overlap_ratio": 0.5,
                "smoothing": 3,
            }
        }
    )

    window = rsc_dialog.ReferenceSpectrumConfigWindow(config_manager, "RSC")
    config = window.get_default_config()

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["reference", "compute", "judgment", "display"]
    assert config["smoothing"] == 3
    assert "octave_smoothing" not in config
    assert config["enable_threshold_judgment"] is True
    assert config["lower_offset_db"] == -2.0
    assert config["upper_offset_db"] == 2.0


def test_excel_dialog_preserves_output_config_after_semantic_migration(qapp):
    from ui.ui_analysis_config.excel_config_dialog import ExcelConfigWindow

    config_manager = FakeConfigManager(
        {
            "Excel": {
                "save_dir": "",
                "file_base": "result",
                "add_date": False,
                "add_model_dir": True,
                "lock_files": False,
                "max_points": 500,
                "save_items": ["SPL", "FBA"],
                "save_mes_enabled": True,
                "mes_file_base": "D:/dataMES",
                "mes_file_name": "MES_Result",
            },
            "SPL": {"type": "SPL"},
            "FBA": {"type": "FBA"},
            "Spec": {"type": "Spec"},
        }
    )

    window = ExcelConfigWindow(config_manager, "Excel")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["output"]
    assert window.get_default_config() == {
        "enabled": True,
        "save_dir": None,
        "file_base": "result",
        "add_date": False,
        "add_model_dir": True,
        "lock_files": False,
        "date_format": "%Y%m%d",
        "max_points": 500,
        "save_items": ["FBA", "SPL"],
        "save_mes_enabled": True,
        "mes_file_base": "D:/dataMES",
        "mes_file_name": "MES_Result",
    }


def test_pd_dialog_preserves_time_smoothing_keys_after_widget_migration(qapp):
    from ui.ui_analysis_config.pd_config_dialog import PDConfigWindow

    config_manager = FakeConfigManager(
        {
            "PD": {
                "smooth_enabled": True,
                "smooth_unit": "points",
                "smooth_time_sec": 0.125,
                "smooth_points": 0,
                "smooth_algo": 3,
            }
        }
    )

    window = PDConfigWindow(config_manager, "PD")
    config = window.get_default_config()

    assert config["smooth_enabled"] is True
    assert config["smooth_unit"] == "points"
    assert config["smooth_time_sec"] == 0.125
    assert config["smooth_points"] == 0
    assert config["smooth_algo"] == 3


def test_pm_dialog_preserves_fixed_threshold_config_after_button_migration(qapp, monkeypatch):
    stub_pattern_match_import_dependencies(monkeypatch)
    from ui.ui_analysis_config import pattern_match_config_dialog as pm_dialog

    monkeypatch.setattr(
        pm_dialog.PatternMatchConfigWindow,
        "load_features_param_config",
        staticmethod(
            lambda: (
                True,
                {
                    "mfcc": {
                        "display_name": "MFCC",
                        "params": {"n_mfcc": {"default": 13}},
                    }
                },
            )
        ),
    )
    config_manager = FakeConfigManager(
        {
            "PM": {
                "pattern_list": [],
                "sample_rate": 48000,
                "feature_type": "mfcc",
                "feature_params": {"n_mfcc": 20},
                "apply_filter": True,
                "filter_range_hz": [100, 5000],
                "similarity_metric": "cosine",
                "threshold_strategy": "fixed_threshold",
                "threshold_value": 0.75,
            }
        }
    )

    window = pm_dialog.PatternMatchConfigWindow(config_manager, "PM")

    assert window.get_config() == {
        "pattern_list": [],
        "sample_rate": 48000,
        "feature_type": "mfcc",
        "feature_params": {"n_mfcc": 20},
        "apply_filter": True,
        "filter_range_hz": (100, 5000),
        "algorithm": "dtw",
        "similarity_metric": "cosine",
        "threshold_strategy": "fixed_threshold",
        "threshold_value": 0.75,
    }


def test_pipeline_pd_pm_dialog_preserves_nested_config_after_base_migration(qapp, monkeypatch):
    stub_pattern_match_import_dependencies(monkeypatch)
    from ui.ui_analysis_config.pipeline_pd_pm_config import PipelinePdPmConfigWindow

    config_manager = FakeConfigManager(
        {
            "ED": {
                "type": "ED",
                "head": {"type": "PD", "config": {"peak_count": 2}},
                "tail": {"type": "PM", "config": {"threshold_value": 0.8}},
                "auto_equal_length": False,
                "left_grid": 10,
                "right_grid": 20,
                "pass_condition": {"n1": 2, "n2": 5},
            }
        }
    )

    window = PipelinePdPmConfigWindow(config_manager, "ED")

    assert window.get_default_config() == {
        "type": "ED",
        "head": {"type": "PD", "config": {"peak_count": 2}},
        "tail": {"type": "PM", "config": {"threshold_value": 0.8}},
        "auto_equal_length": False,
        "left_grid": 10,
        "right_grid": 20,
        "pass_condition": {"n1": 2, "n2": 5},
    }
