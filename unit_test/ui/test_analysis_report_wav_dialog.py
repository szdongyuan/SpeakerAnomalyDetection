from datetime import datetime

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractItemView, QStyle, QStyleOptionViewItem, QWidget

from base.analysis_report_source import ReportCandidate
from consts import ui_style_const
from ui.analysis_report_wav_dialog import (
    AnalysisReportWavDialog,
    CandidateFilterProxyModel,
    CandidateTableModel,
    FilterMultiSelect,
)


def _candidate(
    tmp_path,
    sample,
    port,
    test_round,
    *,
    label="OK",
    database_status="matched",
    condition="0.3",
):
    return ReportCandidate(
        wav_path=str(tmp_path / f"{sample}-{port}.wav"),
        project="ProjectA",
        model="M1",
        sample=sample,
        port=port,
        condition=condition,
        test_round=test_round,
        recorded_at=datetime(2026, 9, 3, 8, 0, test_round),
        label=label,
        database_status=database_status,
    )


def _model(tmp_path):
    model = CandidateTableModel()
    model.reset_index(
        (
            _candidate(tmp_path, "S001", "P1", 1),
            _candidate(tmp_path, "S002", "P2", 2),
        )
    )
    return model


def test_dialog_places_select_all_before_summary_and_uses_final_columns(
    ui_qapp,
    tmp_path,
):
    model = _model(tmp_path)
    parent = QWidget()
    parent.setStyleSheet(ui_style_const.main_window_base_style)
    dialog = AnalysisReportWavDialog(model, parent)
    toolbar_layout = dialog.layout().itemAt(0).layout()
    footer_layout = dialog.layout().itemAt(2).layout()

    headers = [
        dialog.filter_model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        for column in range(dialog.filter_model.columnCount())
    ]
    assert dialog.select_all_checkbox.text() == "全选"
    assert toolbar_layout.itemAt(0).widget() is dialog.select_all_checkbox
    assert toolbar_layout.itemAt(2).widget() is dialog.filter_button
    assert footer_layout.itemAt(0).widget() is dialog.summary_label
    assert footer_layout.itemAt(2).widget() is dialog.cancel_button
    assert footer_layout.itemAt(3).widget() is dialog.confirm_button
    assert dialog.objectName() == "analysisReportWavDialog"
    assert dialog.styleSheet() == ui_style_const.analysis_report_dialog_style
    assert dialog.summary_label.text() == "已选择 2 / 共 2 个 WAV（来自 2 个样本）"
    assert not dialog.candidate_table.alternatingRowColors()
    assert (
        dialog.candidate_table.selectionMode()
        == QAbstractItemView.NoSelection
    )
    dialog.resize(dialog.minimumSize())
    dialog.show()
    ui_qapp.processEvents()
    assert dialog.summary_label.font().pixelSize() == 12
    column_widths = [
        dialog.candidate_table.columnWidth(column)
        for column in range(model.columnCount())
    ]
    business_widths = column_widths[1:CandidateTableModel.RESULT_COLUMN + 1]
    assert column_widths[CandidateTableModel.SELECT_COLUMN] == 54
    assert column_widths[CandidateTableModel.RECORDED_AT_COLUMN] == 190
    assert min(business_widths) >= 85
    assert max(business_widths) - min(business_widths) <= 1
    assert not dialog.candidate_table.horizontalHeader().sectionsClickable()
    assert headers == [
        "选择",
        "样本编号",
        "端口",
        "档位",
        "轮次",
        "总体判定",
        "录制时间",
    ]


def test_candidate_filter_proxy_combines_multi_values_and_tracks_active_filters(
    ui_qapp,
    tmp_path,
):
    model = CandidateTableModel()
    model.reset_index(
        (
            _candidate(tmp_path, "S1", "P1", 1, label="OK"),
            _candidate(tmp_path, "S10", "P2", 2, label="NG"),
            _candidate(tmp_path, "S2", "P1", 2, label="NG"),
            _candidate(tmp_path, "S3", "P3", 3, label=""),
            _candidate(tmp_path, "S4", "P2", 4, label="OK"),
        )
    )
    proxy = CandidateFilterProxyModel()
    proxy.setSourceModel(model)

    assert proxy.filter_values(CandidateTableModel.SAMPLE_COLUMN) == (
        "S1",
        "S2",
        "S3",
        "S4",
        "S10",
    )
    proxy.set_filter_values(
        CandidateTableModel.PORT_COLUMN,
        ("P1", "P2"),
    )
    proxy.set_filter_values(
        CandidateTableModel.RESULT_COLUMN,
        ("NG", "无判定结果"),
    )

    assert proxy.rowCount() == 2
    assert {
        proxy.data(
            proxy.index(row, CandidateTableModel.SAMPLE_COLUMN),
            Qt.DisplayRole,
        )
        for row in range(proxy.rowCount())
    } == {"S2", "S10"}
    assert proxy.has_active_filters()
    assert proxy.active_filter_count() == 2
    assert proxy.active_filters() == {
        CandidateTableModel.PORT_COLUMN: frozenset(("P1", "P2")),
        CandidateTableModel.RESULT_COLUMN: frozenset(
            ("NG", "无判定结果")
        ),
    }

    proxy.set_filter_values(
        CandidateTableModel.PORT_COLUMN,
        proxy.filter_values(CandidateTableModel.PORT_COLUMN),
    )
    assert proxy.active_filter_count() == 1
    assert proxy.rowCount() == 3

    proxy.set_filter_values(CandidateTableModel.RESULT_COLUMN, ())
    assert proxy.active_filter_count() == 1
    assert proxy.rowCount() == 0

    proxy.clear_filters()

    assert proxy.rowCount() == 5
    assert not proxy.has_active_filters()


def test_filter_multiselect_uses_uniform_single_value_state(ui_qapp):
    selector = FilterMultiSelect()
    panel = selector.menu().actions()[0].defaultWidget()
    assert panel.minimumWidth() == 260
    selector.set_values(
        ("P10", "P2", "P1"),
        selected_values=("P1", "P2"),
    )

    assert selector.values() == ("P1", "P2", "P10")
    assert selector.selected_values() == ("P1", "P2")
    assert selector.text() == "已选 2 / 3"
    assert selector.select_all_checkbox.checkState() == Qt.PartiallyChecked

    selector.set_all_checked(True)
    assert selector.selected_values() == ("P1", "P2", "P10")
    assert selector.text() == "ALL"
    assert selector.select_all_checkbox.checkState() == Qt.Checked

    selector.set_all_checked(False)
    assert selector.selected_values() == ()
    assert selector.text() == "未选择"
    assert selector.select_all_checkbox.checkState() == Qt.Unchecked

    selector.set_values(("0.3",), selected_values=("0.3",))
    assert selector.text() == "ALL"
    assert selector.isEnabled()
    assert selector.select_all_checkbox.isEnabled()

    selector.set_all_checked(False)
    assert selector.text() == "未选择"
    assert selector.selected_values() == ()

    selector.set_values(())
    assert selector.text() == "无可选项"
    assert not selector.isEnabled()


def test_dialog_filter_preserves_hidden_selection_and_selects_visible_rows(
    ui_qapp,
    tmp_path,
):
    model = CandidateTableModel()
    model.reset_index(
        (
            _candidate(tmp_path, "S001", "P1", 1, label="OK"),
            _candidate(tmp_path, "S002", "P2", 2, label="NG"),
            _candidate(tmp_path, "S003", "P3", 3, label="OK"),
        )
    )
    dialog = AnalysisReportWavDialog(model)

    dialog.filter_model.set_filter_value(
        CandidateTableModel.RESULT_COLUMN,
        "NG",
    )

    assert dialog.filter_model.rowCount() == 1
    assert dialog.select_all_checkbox.text() == "全选当前筛选结果"
    assert dialog.select_all_checkbox.checkState() == Qt.Checked
    assert dialog.filter_button.text() == "筛选（1）"
    assert dialog.filter_button.property("filtersActive") is True
    assert dialog.summary_label.text() == (
        "已选择 3 / 共 3 个 WAV（当前显示 1 个，来自 3 个样本）"
    )

    dialog.select_all_checkbox.nextCheckState()

    assert [candidate.sample for candidate in model.checked_candidates()] == [
        "S001",
        "S003",
    ]
    assert dialog.select_all_checkbox.checkState() == Qt.Unchecked

    dialog.filter_dialog.reload()
    for selector in dialog.filter_dialog.selector_by_column.values():
        selector.set_all_checked(True)

    assert dialog.filter_model.rowCount() == 1
    dialog.filter_dialog.apply_button.click()

    assert dialog.filter_model.rowCount() == 3
    assert dialog.select_all_checkbox.text() == "全选"
    assert dialog.filter_button.text() == "筛选"
    assert dialog.filter_button.property("filtersActive") is False
    assert [candidate.sample for candidate in model.checked_candidates()] == [
        "S001",
        "S003",
    ]


def test_filter_button_opens_menu_and_keeps_multiple_changes(ui_qapp):
    selector = FilterMultiSelect(all_text="全部样本")
    selector.set_values(("S1", "S2", "S3"))
    selector.show()
    ui_qapp.processEvents()
    observed = []

    def change_options():
        observed.append(selector.menu().isVisible())
        selector.option_list.setCurrentRow(0)
        QTest.keyClick(selector.option_list, Qt.Key_Space)
        selector.option_list.setCurrentRow(1)
        QTest.keyClick(selector.option_list, Qt.Key_Space)
        observed.append(selector.menu().isVisible())
        selector.menu().hide()

    QTimer.singleShot(50, change_options)
    QTest.mouseClick(selector, Qt.LeftButton)
    QTest.qWait(80)
    assert observed == [True, True]
    assert selector.selected_values() == ("S3",)
    assert selector.select_all_checkbox.checkState() == Qt.PartiallyChecked
    selector.close()


def test_filter_dialog_applies_combined_filters_and_restores_values(
    ui_qapp,
    tmp_path,
):
    model = CandidateTableModel()
    model.reset_index(
        (
            _candidate(tmp_path, "S001", "P1", 1, label="OK"),
            _candidate(tmp_path, "S002", "P1", 2, label="NG"),
            _candidate(tmp_path, "S003", "P2", 3, label="NG"),
            _candidate(tmp_path, "S004", "P3", 4, label="NG"),
        )
    )
    dialog = AnalysisReportWavDialog(model)
    filter_dialog = dialog.filter_dialog

    assert filter_dialog.windowTitle() == "WAV 筛选"
    assert filter_dialog.isModal()
    assert filter_dialog.windowFlags() & Qt.WindowType_Mask == Qt.Dialog
    filter_dialog.reload()
    port_selector = filter_dialog.selector_by_column[
        CandidateTableModel.PORT_COLUMN
    ]
    result_selector = filter_dialog.selector_by_column[
        CandidateTableModel.RESULT_COLUMN
    ]
    condition_selector = filter_dialog.selector_by_column[
        CandidateTableModel.CONDITION_COLUMN
    ]
    port_selector.set_selected_values(("P1", "P2"))
    result_selector.set_selected_values(("NG",))

    assert port_selector.text() == "已选 2 / 3"
    assert result_selector.selected_values() == ("NG",)
    assert condition_selector.text() == "全部档位"
    assert condition_selector.isEnabled()

    assert dialog.filter_model.rowCount() == 4
    filter_dialog.apply_button.click()

    assert dialog.filter_model.rowCount() == 2
    assert dialog.filter_button.text() == "筛选（2）"
    assert dialog.filter_button.property("filtersActive") is True
    filter_dialog.reload()
    assert port_selector.selected_values() == ("P1", "P2")
    assert result_selector.selected_values() == ("NG",)
    port_selector.set_all_checked(True)
    result_selector.set_all_checked(True)
    assert dialog.filter_model.rowCount() == 2
    filter_dialog.cancel_button.click()
    assert dialog.filter_model.rowCount() == 2


def test_filter_reset_is_pending_until_apply_and_cancel_restores_filters(
    ui_qapp, tmp_path,
):
    dialog = AnalysisReportWavDialog(_model(tmp_path))
    proxy = dialog.filter_model
    proxy.set_filter_value(CandidateTableModel.PORT_COLUMN, "P1")
    checked_paths = dialog.candidate_model.checked_paths()
    filters = dialog.filter_dialog
    filters.reload()
    filters.reset_button.click()
    assert proxy.rowCount() == 1
    filters.cancel_button.click()
    filters.reload()
    assert filters.selector_by_column[
        CandidateTableModel.PORT_COLUMN
    ].selected_values() == ("P1",)

    filters.reset_button.click()
    filters.apply_button.click()
    assert proxy.rowCount() == 2
    assert not proxy.has_active_filters()
    assert dialog.candidate_model.checked_paths() == checked_paths


def test_result_options_merge_missing_results_and_allow_empty_selection(
    ui_qapp, tmp_path,
):
    model = CandidateTableModel()
    model.reset_index((
        _candidate(tmp_path, "S1", "P1", 1),
        _candidate(tmp_path, "S2", "P2", 2, label="NG"),
        _candidate(tmp_path, "S3", "P3", 3, label=""),
        _candidate(tmp_path, "S4", "P4", 4, label="", database_status="not_found"),
    ))
    dialog = AnalysisReportWavDialog(model)
    filters = dialog.filter_dialog
    filters.reload()
    options = filters.result_options
    assert options.selected_values() == ("OK", "NG", "无判定结果")
    options.set_all_checked(False)
    options.checkboxes["无判定结果"].click()
    filters.apply_button.click()
    assert dialog.filter_model.rowCount() == 2
    assert {
        dialog.filter_model.index(row, CandidateTableModel.SAMPLE_COLUMN).data()
        for row in range(2)
    } == {"S3", "S4"}
    assert [model.candidate_at(row).database_status for row in (2, 3)] == [
        "matched", "not_found",
    ]
    assert "没有有效的 OK / NG" in model.index(2, model.RESULT_COLUMN).data(Qt.ToolTipRole)
    filters.reload()
    assert options.selected_values() == ("无判定结果",)
    options.set_all_checked(False)
    filters.apply_button.click()
    assert dialog.filter_model.rowCount() == 0
    filters.reload()
    assert options.selected_values() == ()

    model.reset_index(())
    filters.reload()
    assert not options.checkboxes
    assert not options.empty_label.isHidden()
    filters.reset_button.click()
    filters.apply_button.click()
    assert dialog.filter_model.rowCount() == 0


def test_filter_layout_fits_minimum_size_with_scoped_font(ui_qapp, tmp_path):
    parent = QWidget()
    parent.setStyleSheet(ui_style_const.main_window_base_style)
    dialog = AnalysisReportWavDialog(_model(tmp_path), parent)
    filters = dialog.filter_dialog
    filters.reload()
    filters.resize(filters.minimumSize())
    filters.show()
    ui_qapp.processEvents()
    selectors = [filters.selector_by_column[column]
                 for _label, column in filters.FILTER_FIELDS[:-1]]
    assert selectors[0].y() == selectors[1].y()
    assert selectors[2].y() == selectors[3].y()
    assert selectors[0].x() == selectors[2].x()
    assert abs(selectors[0].width() - selectors[1].width()) <= 1
    for index, selector in enumerate(selectors):
        row, column = (index // 2) * 3, index % 2
        label = filters.fields_layout.itemAtPosition(row, column).widget()
        assert label.geometry().bottom() < selector.geometry().top()
        if index >= 2:
            assert selectors[index - 2].geometry().bottom() < label.geometry().top()
        assert selector.width() >= selector.minimumSizeHint().width()
        assert selector.font().family() == "Microsoft YaHei UI"
        assert selector.parentWidget().rect().contains(selector.geometry())
    assert filters.result_options.y() > selectors[2].geometry().bottom()
    assert filters.apply_button.parentWidget().rect().contains(
        filters.apply_button.geometry()
    )
    assert dialog.styleSheet() == ui_style_const.analysis_report_dialog_style
    assert parent.styleSheet() == ui_style_const.main_window_base_style
    filters.close()


def test_candidate_result_text_uses_semantic_colors(ui_qapp, tmp_path):
    model = CandidateTableModel()
    model.reset_index(
        (
            _candidate(tmp_path, "S001", "P1", 1, label="OK"),
            _candidate(tmp_path, "S002", "P2", 2, label="NG"),
            _candidate(tmp_path, "S003", "P3", 3, label=""),
            _candidate(
                tmp_path,
                "S004",
                "P4",
                4,
                label="",
                database_status="not_found",
            ),
        )
    )

    colors = [
        model.data(
            model.index(row, CandidateTableModel.RESULT_COLUMN),
            Qt.ForegroundRole,
        ).name().upper()
        for row in range(model.rowCount())
    ]

    assert colors == [
        ui_style_const.COLOR_OK.upper(),
        ui_style_const.COLOR_NG.upper(),
        ui_style_const.COLOR_TEXT_MUTED.upper(),
        ui_style_const.COLOR_TEXT_MUTED.upper(),
    ]
    unjudged_font = model.data(
        model.index(2, CandidateTableModel.RESULT_COLUMN),
        Qt.FontRole,
    )
    assert unjudged_font.pointSizeF() == ui_qapp.font().pointSizeF() - 1.0
    assert (
        model.data(
            model.index(0, CandidateTableModel.RESULT_COLUMN),
            Qt.FontRole,
        )
        is None
    )


def test_dialog_cancel_restores_selection_and_accept_keeps_it(ui_qapp, tmp_path):
    model = _model(tmp_path)
    first = model.index(0, CandidateTableModel.SELECT_COLUMN)

    cancel_dialog = AnalysisReportWavDialog(model)
    assert model.setData(first, Qt.Unchecked, Qt.CheckStateRole)
    assert cancel_dialog.select_all_checkbox.checkState() == Qt.PartiallyChecked
    cancel_dialog.reject()
    assert len(model.checked_candidates()) == 2

    accept_dialog = AnalysisReportWavDialog(model)
    assert model.setData(first, Qt.Unchecked, Qt.CheckStateRole)
    accept_dialog.accept()
    assert [candidate.sample for candidate in model.checked_candidates()] == ["S002"]


def test_dialog_select_all_checkbox_toggles_all_rows(ui_qapp, tmp_path):
    model = _model(tmp_path)
    dialog = AnalysisReportWavDialog(model)

    dialog.select_all_checkbox.nextCheckState()
    assert model.checked_candidates() == []
    assert dialog.select_all_checkbox.checkState() == Qt.Unchecked

    dialog.select_all_checkbox.nextCheckState()
    assert len(model.checked_candidates()) == 2
    assert dialog.select_all_checkbox.checkState() == Qt.Checked


def test_candidate_checkbox_can_be_toggled_by_mouse(ui_qapp, tmp_path):
    model = _model(tmp_path)
    dialog = AnalysisReportWavDialog(model)
    dialog.show()
    ui_qapp.processEvents()

    checkbox_index = dialog.filter_model.index(
        0,
        CandidateTableModel.SELECT_COLUMN,
    )
    option = QStyleOptionViewItem()
    option.rect = dialog.candidate_table.visualRect(checkbox_index)
    dialog.candidate_table.itemDelegate(checkbox_index).initStyleOption(
        option,
        checkbox_index,
    )
    click_position = dialog.candidate_table.style().subElementRect(
        QStyle.SE_ItemViewItemCheckIndicator,
        option,
        dialog.candidate_table,
    ).center()

    QTest.mouseClick(
        dialog.candidate_table.viewport(),
        Qt.LeftButton,
        pos=click_position,
    )
    ui_qapp.processEvents()

    assert [candidate.sample for candidate in model.checked_candidates()] == ["S002"]
    assert dialog.select_all_checkbox.checkState() == Qt.PartiallyChecked
    dialog.reject()
