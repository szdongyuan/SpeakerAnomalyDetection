from datetime import datetime
from dataclasses import replace
from threading import Event
from time import monotonic

import pytest
from PyQt5.QtCore import QEvent, QObject, QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QLabel, QToolTip

from base.analysis_report_source import (
    AnalysisItemIdentity,
    CandidateAnalysisItem,
    ProjectReportIndex,
    ReportCandidate,
)
from consts import ui_style_const
from ui.analysis_report_export_dialog import (
    AnalysisReportExportDialog,
    SampleMultiSelect,
)
from ui.analysis_report_wav_dialog import CandidateTableModel


SPL_IDENTITY = AnalysisItemIdentity("低频SPL", "SPL")
FFT_IDENTITY = AnalysisItemIdentity("全频FFT", "FFT")


def test_database_warning_remains_separate_from_missing_results(ui_qapp, tmp_path):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    index = _index(tmp_path)
    dialog.load_index(replace(index, warnings=("数据库文件不存在",)))
    assert not dialog.database_warning_label.isHidden()
    assert "判定数据读取失败" in dialog.database_warning_label.text()
    assert "数据库文件不存在" in dialog.database_warning_label.text()
    dialog._set_busy(False)
    assert not dialog.database_warning_label.isHidden()
    dialog.load_index(index)
    assert dialog.database_warning_label.isHidden()
    assert not dialog.database_warning_label.text()
    dialog.load_index(replace(index, warnings=("数据库文件不存在",)))
    dialog._clear_loaded_index()
    assert dialog.database_warning_label.isHidden()
    dialog.close()


@pytest.mark.parametrize("activity", ["scan", "export"])
def test_escape_cancels_active_task_without_closing_dialog(ui_qapp, tmp_path, monkeypatch, activity):
    from base.analysis_report import AnalysisReportExportResult

    release = Event()
    messages = []

    def scan(*args, **kwargs):
        release.wait(3)
        return _index(tmp_path)

    def export(*args, **kwargs):
        release.wait(3)
        return AnalysisReportExportResult(False, "已取消", cancelled=kwargs["cancel_requested"]())

    module = "ui.analysis_report_export_dialog"
    monkeypatch.setattr(f"{module}.scan_project", scan)
    monkeypatch.setattr(f"{module}.export_analysis_report_pdf", export)
    monkeypatch.setattr(f"{module}.prepare_analysis_report_runtime", lambda: None)
    monkeypatch.setattr(f"{module}.QFileDialog.getSaveFileName",
                        lambda *args: (str(tmp_path / "report.pdf"), ""))
    monkeypatch.setattr(f"{module}.QMessageBox.critical", lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(f"{module}.QMessageBox.information", lambda *args: messages.append(args[-1]))
    dialog = AnalysisReportExportDialog(project_directory=str(tmp_path), auto_scan=False)
    rejected = QSignalSpy(dialog.rejected)
    dialog.show()
    thread = None
    try:
        if activity == "scan":
            dialog._start_scan()
            thread = dialog._scan_thread
        else:
            dialog.load_index(_index(tmp_path))
            dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
            dialog._start_export()
            thread = dialog._export_thread
        assert thread is not None and dialog._busy
        for _ in range(2):
            QTest.keyClick(dialog, Qt.Key_Escape)
            ui_qapp.processEvents()
            assert dialog.isVisible() and not rejected
            assert thread.isInterruptionRequested()
            assert "正在取消" in dialog.status_label.text()
            assert not dialog.cancel_button.isEnabled()
        release.set()
        deadline = monotonic() + 3
        while dialog._busy and monotonic() < deadline:
            QTest.qWait(10)
        assert not dialog._busy and dialog.isVisible()
        assert dialog.browse_button.isEnabled() and dialog.export_button.isEnabled()
        assert dialog.cancel_button.isEnabled() and dialog.cancel_button.text() == "关闭"
        assert not messages
        if activity == "scan":
            assert dialog.candidate_model.rowCount() == 0
        else:
            assert dialog.candidate_model.rowCount() == 2
            assert dialog.analysis_item_panel.selected_items() == [SPL_IDENTITY]
        QTest.keyClick(dialog, Qt.Key_Escape)
        assert not dialog.isVisible() and len(rejected) == 1
    finally:
        release.set()
        active_thread = dialog._scan_thread or dialog._export_thread
        if active_thread is not None:
            active_thread.wait(3000)
            ui_qapp.processEvents()
        dialog.close()


@pytest.mark.parametrize("outcome", ["loaded", "empty", "failed", "cancelled"])
def test_scan_feedback_stays_responsive_and_clears_after_finish(ui_qapp, tmp_path, monkeypatch, outcome):
    release = Event()
    errors = []

    def scan(*args, **kwargs):
        assert release.wait(3), "test did not release the scan worker"
        if outcome == "failed":
            raise OSError("目录无法读取")
        if outcome == "empty":
            return ProjectReportIndex(str(tmp_path), "ProjectA", ())
        # Also exercise a result arriving just after cancellation was requested.
        return _index(tmp_path)

    monkeypatch.setattr("ui.analysis_report_export_dialog.scan_project", scan)
    monkeypatch.setattr("ui.analysis_report_export_dialog.QMessageBox.critical",
                        lambda *args: errors.append(args[-1]))
    dialog = AnalysisReportExportDialog(project_directory=str(tmp_path), auto_scan=False)
    dialog.show()
    try:
        dialog._start_scan()
        painted = []
        QTimer.singleShot(0, lambda: painted.append(True))
        ui_qapp.processEvents()
        assert painted and dialog._busy
        assert dialog.progress_bar.isVisible()
        assert dialog.progress_bar.maximum() == 0
        assert "正在扫描" in dialog.status_label.text()
        assert dialog.analysis_item_panel._empty_label.text() == ""
        assert not dialog.browse_button.isEnabled()
        assert not dialog.export_button.isEnabled()
        if outcome == "cancelled":
            dialog._cancel_or_close()
            assert "正在取消" in dialog.status_label.text()
        release.set()
        deadline = monotonic() + 3
        while dialog._busy and monotonic() < deadline:
            QTest.qWait(10)
        assert not dialog._busy
        assert dialog.progress_bar.isHidden() and dialog.status_label.isHidden()
        assert dialog.browse_button.isEnabled()
        assert dialog.cancel_button.text() == "关闭"
        if outcome == "loaded":
            assert dialog.model_combo.currentData() == "M1"
            assert dialog.candidate_model.rowCount() == 2
        else:
            expected = {"empty": "未发现可导出的 WAV", "failed": "扫描失败", "cancelled": "已取消扫描"}
            assert expected[outcome] in dialog.analysis_item_panel._empty_label.text()
            assert dialog.candidate_model.rowCount() == 0
        assert errors == (["目录无法读取"] if outcome == "failed" else [])
    finally:
        release.set()
        if dialog._scan_thread is not None:
            dialog._scan_thread.wait(3000)
            ui_qapp.processEvents()
        dialog.close()


def _candidate(
    path,
    *,
    model,
    sample,
    port,
    items,
    condition="0.3",
    test_round=1,
):
    return ReportCandidate(
        wav_path=str(path),
        project="ProjectA",
        model=model,
        sample=sample,
        port=port,
        condition=condition,
        test_round=test_round,
        recorded_at=datetime(2026, 9, 2, 8, 0, test_round),
        label="OK",
        database_status="matched",
        analysis_items=tuple(items),
        channel_labels=(("CH1", "前"),),
        channel_mapping_source="录制时通道映射快照",
    )


def _index(tmp_path):
    spl = CandidateAnalysisItem(
        SPL_IDENTITY,
        csv_files=(("总体声压级", str(tmp_path / "spl.csv")),),
    )
    fft = CandidateAnalysisItem(
        FFT_IDENTITY,
        image_files=((1, str(tmp_path / "fft.png")),),
    )
    return ProjectReportIndex(
        project_directory=str(tmp_path),
        project_name="ProjectA",
        candidates=(
            _candidate(
                tmp_path / "m1-s001.wav",
                model="M1",
                sample="S001",
                port="P1",
                items=(spl, fft),
            ),
            _candidate(
                tmp_path / "m1-s002.wav",
                model="M1",
                sample="S002",
                port="P2",
                items=(spl,),
                test_round=2,
            ),
            _candidate(
                tmp_path / "m2-s001.wav",
                model="M2",
                sample="S001",
                port="P3",
                items=(fft,),
            ),
        ),
    )


def _select_model(dialog, model):
    index = dialog.model_combo.findData(model)
    assert index >= 0
    dialog.model_combo.setCurrentIndex(index)


def _select_samples(dialog, *samples):
    dialog.sample_selector.set_selected_values(samples)


def test_loaded_index_defaults_to_first_model_and_selects_all_samples(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))

    assert [dialog.model_combo.itemText(row) for row in range(2)] == [
        "M1",
        "M2",
    ]
    assert dialog.model_combo.currentData() == "M1"
    assert dialog.sample_selector.isEnabled()
    assert dialog.sample_selector.option_count() == 2
    assert dialog.sample_selector.selected_values() == ["S001", "S002"]
    assert dialog.sample_selector.text() == "全部已选（2）"
    assert dialog.candidate_model.rowCount() == 2
    assert dialog.analysis_item_panel.option_count() == 2
    assert dialog.export_button.isEnabled()
    assert dialog.export_button.text() == "导出 PDF…"
    assert dialog.analysis_item_group.title() == "导出内容"
    assert not dialog.analysis_item_label.isHidden()
    assert dialog.values_only_radio.isChecked()
    assert not dialog.values_only_radio.isEnabled()
    assert not dialog.values_and_charts_radio.isEnabled()
    assert all(
        label.text() != "图表："
        for label in dialog.findChildren(QLabel)
    )
    assert dialog.model_combo.maximumWidth() > 320
    assert dialog.sample_selector.maximumWidth() > 320
    assert not hasattr(dialog, "scan_button")
    assert dialog.progress_bar.isHidden()
    assert dialog.status_label.isHidden()
    assert not hasattr(dialog, "port_selector")
    assert not hasattr(dialog, "analysis_item_table")


def test_loaded_index_without_models_shows_disabled_empty_state(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(
        ProjectReportIndex(
            project_directory=str(tmp_path),
            project_name="ProjectA",
            candidates=(),
        )
    )

    assert dialog.model_combo.count() == 1
    assert dialog.model_combo.currentText() == "未发现型号"
    assert dialog.model_combo.currentData() == ""
    assert not dialog.model_combo.isEnabled()
    assert not dialog.sample_selector.isEnabled()


def test_initial_layout_has_balanced_spacing_and_keeps_fields_near_labels(
    ui_qapp,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.show()
    ui_qapp.processEvents()

    assert 460 <= dialog.height() <= 490
    assert dialog.objectName() == "analysisReportExportDialog"
    assert dialog.styleSheet() == ui_style_const.analysis_report_dialog_style
    root_margins = dialog.layout().contentsMargins()
    assert (
        root_margins.left(),
        root_margins.top(),
        root_margins.right(),
        root_margins.bottom(),
    ) == (20, 20, 20, 20)
    assert dialog.layout().spacing() == 16
    assert dialog.status_label.isHidden()
    assert dialog.project_path_edit.placeholderText() == "请选择项目目录"
    assert not dialog.analysis_item_label.isHidden()
    assert dialog.analysis_item_panel._empty_label.text() == "请选择项目目录以加载分析项"
    field_left_edges = [
        widget.mapTo(dialog, QPoint(0, 0)).x()
        for widget in (
            dialog.project_path_edit,
            dialog.model_combo,
            dialog.sample_selector,
        )
    ]
    assert max(field_left_edges) - min(field_left_edges) <= 1
    empty_state_left = dialog.analysis_item_panel._empty_label.mapTo(
        dialog,
        QPoint(0, 0),
    ).x()
    assert abs(empty_state_left - field_left_edges[0]) <= 1
    assert dialog.analysis_item_panel.height() >= 24
    assert dialog.analysis_item_panel._empty_label.alignment() == (Qt.AlignLeft | Qt.AlignVCenter)
    assert not dialog.report_content_controls.isVisible()
    assert dialog.report_content_label.geometry().top() > dialog.analysis_item_panel.geometry().bottom()
    assert (
        dialog.analysis_item_panel.mapTo(dialog, QPoint(0, 0)).y()
        - dialog.analysis_item_group.y()
        <= 50
    )
    assert dialog.sample_selector.width() == dialog.model_combo.width()
    assert dialog.details_button.x() - dialog.sample_selector.geometry().right() <= 12
    assert 185 <= dialog.scope_group.height() <= 220
    assert 160 <= dialog.analysis_item_group.height() <= 210
    assert (
        dialog.analysis_item_group.y()
        - dialog.scope_group.geometry().bottom()
        >= 14
    )
    image = dialog.grab().toImage()
    group_fill = image.pixelColor(
        dialog.width() - 40,
        dialog.scope_group.y() + dialog.scope_group.height() // 2,
    )
    assert group_fill.name().upper() == ui_style_const.COLOR_CARD_BG.upper()
    group_left_mid = dialog.scope_group.mapTo(
        dialog,
        QPoint(0, dialog.scope_group.height() // 2),
    )
    assert (
        image.pixelColor(group_left_mid).name().upper()
        == ui_style_const.COLOR_BORDER.upper()
    )
    group_title_rule = dialog.scope_group.mapTo(
        dialog,
        QPoint(dialog.scope_group.width() // 2, 10),
    )
    assert (
        image.pixelColor(group_title_rule).name().upper()
        == ui_style_const.COLOR_CARD_BG.upper()
    )
    disabled_model_fill = dialog.model_combo.mapTo(
        dialog,
        QPoint(dialog.model_combo.width() - 50, dialog.model_combo.height() // 2),
    )
    assert (
        image.pixelColor(disabled_model_fill).name().upper()
        == ui_style_const.COLOR_CARD_BG.upper()
    )

    dialog.resize(dialog.minimumWidth(), dialog.minimumHeight())
    ui_qapp.processEvents()
    assert dialog.details_button.geometry().right() < dialog.scope_group.geometry().right()


def test_model_populates_and_selects_all_samples(ui_qapp, tmp_path):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))

    _select_model(dialog, "M1")

    assert dialog.sample_selector.isEnabled()
    assert [
        dialog.sample_selector.option_list.item(row).text()
        for row in range(dialog.sample_selector.option_count())
    ] == ["S001", "S002"]
    assert dialog.sample_selector.selected_values() == ["S001", "S002"]
    assert dialog.sample_selector.text() == "全部已选（2）"
    assert dialog.candidate_model.rowCount() == 2
    assert not hasattr(dialog.sample_selector, "search_edit")


def test_sample_selector_natural_sort_tri_state_and_single_emission(ui_qapp):
    selector = SampleMultiSelect()
    selector.set_values(("S10", "S2", "S1"))
    spy = QSignalSpy(selector.selection_changed)

    assert [
        selector.option_list.item(row).text()
        for row in range(selector.option_count())
    ] == ["S1", "S2", "S10"]
    assert selector.select_all_checkbox.text() == "全选"
    assert selector.select_all_checkbox.checkState() == Qt.Unchecked
    assert selector.text() == "未选择"

    selector.set_all_checked(True)
    assert len(spy) == 1
    assert selector.selected_values() == ["S1", "S2", "S10"]
    assert selector.select_all_checkbox.checkState() == Qt.Checked
    assert selector.text() == "全部已选（3）"

    selector.option_list.item(0).setCheckState(Qt.Unchecked)
    assert len(spy) == 2
    assert selector.select_all_checkbox.checkState() == Qt.PartiallyChecked
    assert selector.text() == "已选 2 / 3"

    selector.select_all_checkbox.nextCheckState()
    assert len(spy) == 3
    assert selector.select_all_checkbox.checkState() == Qt.Checked

    selector.select_all_checkbox.nextCheckState()
    assert len(spy) == 4
    assert selector.selected_values() == []
    assert selector.select_all_checkbox.checkState() == Qt.Unchecked
    assert selector.text() == "未选择"


def test_selected_samples_define_and_select_exact_wav_candidates(ui_qapp, tmp_path):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")

    _select_samples(dialog, "S001")

    selected = dialog.candidate_model.checked_candidates()
    assert dialog.candidate_model.rowCount() == 1
    assert [(candidate.model, candidate.sample) for candidate in selected] == [
        ("M1", "S001")
    ]
    assert dialog.analysis_item_panel.option_count() == 2
    assert dialog.analysis_item_panel.selected_items() == []
    assert dialog.export_button.isEnabled()


def test_switching_model_selects_new_samples_wavs_and_resets_analysis_items(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001")
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)

    _select_model(dialog, "M2")

    assert dialog.sample_selector.selected_values() == ["S001"]
    assert dialog.sample_selector.option_count() == 1
    assert dialog.candidate_model.rowCount() == 1
    assert dialog.analysis_item_panel.option_count() == 1
    assert dialog.analysis_item_panel.selected_items() == []
    assert dialog.export_button.isEnabled()


def test_clearing_default_sample_selection_clears_wavs_and_analysis_items(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))

    dialog.sample_selector.set_all_checked(False)

    assert dialog.sample_selector.selected_values() == []
    assert dialog.sample_selector.text() == "未选择"
    assert dialog.candidate_model.rowCount() == 0
    assert dialog.analysis_item_panel.option_count() == 0


def test_analysis_items_are_direct_exact_checkboxes_with_partial_coverage_tooltip(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")

    assert dialog.analysis_item_panel.option_count() == 2
    assert all(
        "/" not in checkbox.text()
        for checkbox in dialog.analysis_item_panel._checkboxes.values()
    )
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)

    assert dialog.analysis_item_panel.selected_items() == [FFT_IDENTITY]
    assert len(dialog.candidate_model.checked_candidates()) == 2
    checkbox = dialog.analysis_item_panel._checkboxes[FFT_IDENTITY]
    assert checkbox.toolTip() == "部分 WAV 缺少该分析项结果。"
    assert not hasattr(dialog, "missing_summary_label")
    assert dialog.export_button.isEnabled()


def test_wav_details_use_separate_dialog_and_recording_time_is_last(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001")

    headers = [
        dialog.candidate_model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        for column in range(dialog.candidate_model.columnCount())
    ]
    assert not hasattr(dialog, "detail_container")
    assert not hasattr(dialog, "candidate_table")
    dialog.show()
    ui_qapp.processEvents()
    assert dialog.details_button.text() == "查看 WAV"
    assert dialog.details_button.toolTip() == "查看并调整已匹配的 WAV 明细"
    assert (
        dialog.details_button.x() - dialog.sample_selector.geometry().right() - 1
        >= 8
    )
    assert headers == [
        "选择",
        "样本编号",
        "端口",
        "档位",
        "轮次",
        "总体判定",
        "录制时间",
    ]
    assert headers[-1] == "录制时间"
    assert "WAV 文件名" not in headers
    assert "数据状态" not in headers
    tooltip = dialog.candidate_model.data(
        dialog.candidate_model.index(0, 1),
        Qt.ToolTipRole,
    )
    assert "m1-s001.wav" in tooltip
    assert str(tmp_path / "m1-s001.wav") in tooltip


def test_wav_details_button_opens_separate_dialog(
    ui_qapp,
    tmp_path,
    monkeypatch,
):
    opened = []

    class FakeWavDialog:
        def __init__(self, candidate_model, parent):
            opened.append((candidate_model, parent))

        @staticmethod
        def exec():
            return 0

        @staticmethod
        def deleteLater():
            return None

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.AnalysisReportWavDialog",
        FakeWavDialog,
    )
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001")

    dialog.details_button.click()

    assert opened == [(dialog.candidate_model, dialog)]


def test_cancelled_wav_adjustment_does_not_clear_selected_analysis_item(
    ui_qapp,
    tmp_path,
    monkeypatch,
):
    class FakeCancelledWavDialog:
        def __init__(self, candidate_model, _parent):
            self.candidate_model = candidate_model
            self.initial_paths = candidate_model.checked_paths()

        def exec(self):
            self.candidate_model.setData(
                self.candidate_model.index(0, CandidateTableModel.SELECT_COLUMN),
                Qt.Unchecked,
                Qt.CheckStateRole,
            )
            self.candidate_model.set_checked_paths(self.initial_paths)
            return 0

        @staticmethod
        def deleteLater():
            return None

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.AnalysisReportWavDialog",
        FakeCancelledWavDialog,
    )
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)

    dialog._open_wav_details()

    assert dialog.analysis_item_panel.selected_items() == [FFT_IDENTITY]
    assert len(dialog.candidate_model.checked_candidates()) == 2


def test_confirmed_wav_adjustment_refreshes_available_analysis_items(
    ui_qapp,
    tmp_path,
    monkeypatch,
):
    class FakeConfirmedWavDialog:
        def __init__(self, candidate_model, _parent):
            self.candidate_model = candidate_model

        def exec(self):
            self.candidate_model.setData(
                self.candidate_model.index(0, CandidateTableModel.SELECT_COLUMN),
                Qt.Unchecked,
                Qt.CheckStateRole,
            )
            return 1

        @staticmethod
        def deleteLater():
            return None

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.AnalysisReportWavDialog",
        FakeConfirmedWavDialog,
    )
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)

    dialog._open_wav_details()

    assert [
        candidate.sample for candidate in dialog.candidate_model.checked_candidates()
    ] == ["S002"]
    assert dialog.analysis_item_panel.selected_items() == []
    assert dialog.analysis_item_panel.option_count() == 1


@pytest.mark.parametrize("minimum_size", [False, True])
def test_report_content_appears_only_after_selection_without_resizing(ui_qapp, tmp_path, minimum_size):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.show()
    ui_qapp.processEvents()
    if minimum_size:
        dialog.resize(dialog.minimumSize())
        ui_qapp.processEvents()
    initial_size = dialog.size()
    footer_position = dialog.export_button.mapTo(dialog, QPoint())

    def assert_report_visible(visible):
        ui_qapp.processEvents()
        for widget in (
            dialog.report_content_label,
            dialog.values_only_radio,
            dialog.values_and_charts_radio,
        ):
            assert widget.isVisible() == visible
        assert not dialog.report_content_hint.isVisible()
        assert dialog.size() == initial_size
        assert dialog.export_button.mapTo(dialog, QPoint()) == footer_position

    try:
        assert_report_visible(False)
        dialog.load_index(_index(tmp_path))
        assert_report_visible(False)
        dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
        assert_report_visible(True)
        dialog.analysis_item_panel.set_checked(SPL_IDENTITY, False)
        assert_report_visible(False)
        dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
        assert_report_visible(True)
        assert dialog.values_and_charts_radio.isChecked()
        dialog.sample_selector.set_selected_values(())
        assert_report_visible(False)
        assert dialog.analysis_item_panel._empty_label.text() == "请先选择样本"
        dialog.sample_selector.set_all_checked(True)
        dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
        assert_report_visible(True)
        dialog.candidate_model.set_checked_paths(())
        assert_report_visible(False)
        dialog._clear_loaded_index()
        assert_report_visible(False)
        assert dialog.analysis_item_panel._empty_label.text() == "请选择项目目录以加载分析项"
    finally:
        dialog.close()


def test_report_content_modes_follow_available_results_and_stay_exclusive(
    ui_qapp,
    tmp_path,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")

    assert dialog._report_content() == "values_only"
    assert not dialog.values_only_radio.isEnabled()
    assert not dialog.values_and_charts_radio.isEnabled()

    assert dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
    assert dialog.values_only_radio.isEnabled()
    assert not dialog.values_and_charts_radio.isEnabled()

    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    assert not dialog.values_only_radio.isEnabled()
    assert dialog.values_and_charts_radio.isEnabled()
    assert "全频FFT" in dialog.values_only_radio.toolTip()
    assert "取消勾选" in dialog.values_only_radio.toolTip()
    assert dialog.report_content_hint.isHidden()
    assert dialog._report_content() == "values_and_charts"
    assert not dialog.values_only_radio.isChecked()
    dialog.values_only_radio.click()
    assert dialog._report_content() == "values_and_charts"

    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, False)
    assert not dialog.values_and_charts_radio.isEnabled()
    assert not dialog.values_and_charts_radio.isChecked()
    assert dialog._report_content() == "values_only"

    dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    dialog.analysis_item_panel.set_checked(SPL_IDENTITY, False)
    assert dialog._report_content() == "values_and_charts"
    assert not dialog.values_only_radio.isEnabled()
    assert "全频FFT" in dialog.values_only_radio.toolTip()
    assert dialog.report_content_hint.isHidden()
    assert not dialog.values_only_radio.isChecked()
    dialog._set_busy(True)
    assert not dialog.values_and_charts_radio.isEnabled()
    dialog._set_busy(False)
    assert dialog.values_and_charts_radio.isEnabled()
    assert not dialog.values_only_radio.isEnabled()
    dialog._clear_loaded_index()
    assert dialog._report_content() == "values_only"
    assert not dialog.values_and_charts_radio.isEnabled()
    assert not dialog.values_only_radio.toolTip()
    assert not dialog.values_and_charts_radio.toolTip()


@pytest.mark.parametrize("analysis_type", ["FBA", "FFT", "Spec"])
@pytest.mark.parametrize("has_chart_file", [True, False])
def test_mixed_items_require_charts_until_chart_item_is_removed(
    ui_qapp, tmp_path, analysis_type, has_chart_file,
):
    index = _index(tmp_path)
    identity = AnalysisItemIdentity(f"测试{analysis_type}", analysis_type)
    scalar_item = replace(
        index.candidates[0].analysis_items[0],
        image_files=((1, str(tmp_path / "spl.png")),),
    )
    chart_item = CandidateAnalysisItem(
        identity,
        image_files=((1, str(tmp_path / "chart.png")),) if has_chart_file else (),
    )
    candidate = replace(index.candidates[0], analysis_items=(scalar_item, chart_item))
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(replace(index, candidates=(candidate,)))
    try:
        dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
        assert dialog.values_only_radio.isEnabled()
        assert dialog.values_and_charts_radio.isEnabled()
        dialog.values_and_charts_radio.click()
        assert dialog._report_content() == "values_and_charts"
        dialog.values_only_radio.click()
        assert dialog._report_content() == "values_only"

        dialog.analysis_item_panel.set_checked(identity, True)
        assert dialog._report_content() == "values_and_charts"
        assert not dialog.values_only_radio.isEnabled()
        assert identity.key in dialog.values_only_radio.toolTip()
        dialog.values_only_radio.click()
        assert dialog._report_content() == "values_and_charts"
        assert dialog.report_content_hint.isHidden() == has_chart_file
        if not has_chart_file:
            assert identity.key in dialog.report_content_hint.text()
            assert "缺少分析图" in dialog.report_content_hint.text()
        dialog._set_busy(True)
        dialog._set_busy(False)
        assert not dialog.values_only_radio.isEnabled()

        dialog.analysis_item_panel.set_checked(identity, False)
        assert dialog.values_only_radio.isEnabled()
        assert dialog.values_and_charts_radio.isEnabled()
        dialog.values_only_radio.click()
        assert dialog._report_content() == "values_only"
        assert not dialog.values_only_radio.toolTip()
        assert dialog.report_content_hint.isHidden()
    finally:
        dialog.close()


@pytest.mark.parametrize("identity,control,expected", [
    (FFT_IDENTITY, "values_only_radio", "已选分析项（全频FFT）需要包含分析图；如需仅导出数值，请取消勾选这些项。"),
    (SPL_IDENTITY, "values_and_charts_radio", "当前所选结果没有可导出的分析图。"),
])
def test_unavailable_report_mode_shows_tooltip(ui_qapp, tmp_path, identity, control, expected):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    dialog.analysis_item_panel.set_checked(identity, True)
    dialog.show()
    dialog.activateWindow()
    QTest.qWait(100)
    ui_qapp.processEvents()
    radio = getattr(dialog, control)
    try:
        assert not radio.isEnabled()
        assert not dialog.report_content_hint.isVisible()
        position = radio.rect().center()
        ui_qapp.sendEvent(radio, QHelpEvent(QEvent.ToolTip, position, radio.mapToGlobal(position)))
        QTest.qWait(250)
        assert QToolTip.isVisible()
        assert QToolTip.text() == expected
    finally:
        QToolTip.hideText()
        dialog.close()


@pytest.mark.parametrize("key,modifiers", [
    (Qt.Key_Return, Qt.NoModifier),
    (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_chart_only_enter_opens_save_in_chart_mode(ui_qapp, tmp_path, monkeypatch, key, modifiers):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    calls = []
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QFileDialog.getSaveFileName",
        lambda *args: (calls.append(dialog._report_content()) or "", ""),
    )
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QMessageBox.warning",
        lambda *args: pytest.fail(args[-1]),
    )
    dialog.show()
    ui_qapp.processEvents()
    dialog.values_and_charts_radio.setFocus()
    QTest.keyClick(dialog.values_and_charts_radio, key, modifiers)
    assert calls == ["values_and_charts"]
    assert dialog._export_thread is None
    dialog.close()


def test_multiple_analysis_rows_remain_readable_at_minimum_size(ui_qapp, tmp_path):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    index = _index(tmp_path)
    dialog.load_index(index)
    dialog.show()
    ui_qapp.processEvents()
    items = tuple(
        CandidateAnalysisItem(
            AnalysisItemIdentity(f"声压级通道组{number}", "SPL"),
            csv_files=(("总体声压级", str(tmp_path / f"{number}.csv")),),
        )
        for number in range(8)
    )
    dialog.load_index(replace(index, candidates=(replace(index.candidates[0], analysis_items=items),)))
    dialog.analysis_item_panel.set_checked(items[0].identity, True)
    ui_qapp.processEvents()
    dialog.resize(dialog.minimumSize())
    ui_qapp.processEvents()
    checkboxes = list(dialog.analysis_item_panel._checkboxes.values())
    for checkbox in checkboxes:
        assert checkbox.height() >= checkbox.sizeHint().height()
        assert checkbox.width() >= checkbox.sizeHint().width()
    assert checkboxes[0].y() == checkboxes[1].y()
    assert checkboxes[2].y() > checkboxes[0].geometry().bottom()
    assert checkboxes[0].mapTo(dialog, QPoint()).x() == dialog.values_only_radio.mapTo(dialog, QPoint()).x()
    assert checkboxes[1].mapTo(dialog, QPoint()).x() == dialog.values_and_charts_radio.mapTo(dialog, QPoint()).x()
    assert dialog.project_path_edit.height() >= dialog.project_path_edit.minimumSizeHint().height()
    assert dialog.export_button.geometry().bottom() < dialog.height()
    dialog.close()


def test_export_button_stays_actionable_and_validates_missing_project(
    ui_qapp,
    monkeypatch,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    warnings = []
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QMessageBox.warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert dialog.export_button.isEnabled()
    dialog.export_button.click()

    assert warnings == [("报告导出", "请先选择项目目录。")]


def test_export_rejects_analysis_item_without_scalar_values_or_images(
    ui_qapp,
    tmp_path,
    monkeypatch,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    index = _index(tmp_path)
    candidate = replace(index.candidates[-1], analysis_items=(
        CandidateAnalysisItem(FFT_IDENTITY, csv_files=(("FFT", str(tmp_path / "fft.csv")),)),
    ))
    dialog.load_index(replace(index, candidates=(candidate,)))
    _select_model(dialog, "M2")
    _select_samples(dialog, "S001")
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    assert not dialog.values_only_radio.isEnabled()
    assert not dialog.values_and_charts_radio.isEnabled()
    assert not dialog.report_content_hint.isHidden()
    assert "暂无可导出" in dialog.report_content_hint.text()
    warnings = []

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QMessageBox.warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("数值模式校验失败后不应打开保存对话框")
        ),
    )

    dialog._start_export()

    assert warnings == [
        (
            "报告导出",
            "当前所选分析项没有可导出的数值或分析图，请选择其他分析项。",
        )
    ]
    dialog.analysis_item_panel.set_checked(FFT_IDENTITY, False)
    assert dialog.report_content_hint.isHidden()
    assert not dialog.report_content_hint.text()


def test_incomplete_items_go_directly_to_save_dialog_without_deselecting_wavs(
    ui_qapp,
    tmp_path,
    monkeypatch,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    assert dialog.values_and_charts_radio.isChecked()
    save_dialog_calls = []

    def save_dialog(*args, **kwargs):
        save_dialog_calls.append((args, kwargs))
        return "", ""

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QMessageBox.question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("缺失分析项不应触发额外确认对话框")
        ),
    )
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QFileDialog.getSaveFileName",
        save_dialog,
    )

    dialog._start_export()

    assert len(save_dialog_calls) == 1
    assert len(dialog.candidate_model.checked_candidates()) == 2


@pytest.mark.parametrize("include_scalar_item", [False, True])
def test_export_passes_exact_scope_to_export_thread(
    ui_qapp,
    tmp_path,
    monkeypatch,
    include_scalar_item,
):
    dialog = AnalysisReportExportDialog(auto_scan=False)
    dialog.load_index(_index(tmp_path))
    _select_model(dialog, "M1")
    _select_samples(dialog, "S001", "S002")
    if include_scalar_item:
        dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
    assert dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    assert dialog.values_and_charts_radio.isChecked()
    captured = {}

    class FakeExportThread(QObject):
        completed = pyqtSignal(object)
        finished = pyqtSignal()

        def __init__(
            self,
            file_path,
            candidates,
            analysis_items,
            report_content,
            parent=None,
        ):
            super().__init__(parent)
            captured.update(
                file_path=file_path,
                candidates=tuple(candidates),
                analysis_items=tuple(analysis_items),
                report_content=report_content,
            )

        def start(self):
            captured["started"] = True

        def requestInterruption(self):
            captured["interrupted"] = True

    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(tmp_path / "report.pdf"), "PDF 文件 (*.pdf)"),
    )
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog.prepare_analysis_report_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "ui.analysis_report_export_dialog._ReportExportThread",
        FakeExportThread,
    )

    dialog._start_export()

    assert captured["started"] is True
    assert captured["file_path"] == str(tmp_path / "report.pdf")
    assert [candidate.sample for candidate in captured["candidates"]] == [
        "S001",
        "S002",
    ]
    expected_items = (SPL_IDENTITY, FFT_IDENTITY) if include_scalar_item else (FFT_IDENTITY,)
    assert set(captured["analysis_items"]) == set(expected_items)
    assert captured["report_content"] == "values_and_charts"
    assert not dialog.export_button.isEnabled()
    assert dialog.cancel_button.text() == "取消任务"
    dialog._set_busy(False)
    assert dialog.export_button.isEnabled()
    assert dialog.cancel_button.text() == "关闭"
