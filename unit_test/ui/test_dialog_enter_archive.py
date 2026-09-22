"""Real Enter-key integration for report, archive and result dialogs."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QPushButton

from base.analysis_process_protocol import AnalysisSegmentResult, AnalysisTaskResult
from base.analysis_report_source import ProjectReportIndex
from base.analysis_segments import AnalysisSegment
from base.audio_record_filter import parse_audio_filter_metadata
from consts import error_code
from ui.analysis_multichannel_result_window import AnalysisMultichannelResultWindow
from ui.analysis_report_export_dialog import AnalysisReportExportDialog
from ui.analysis_report_wav_dialog import AnalysisReportWavDialog, CandidateTableModel
from ui.archive_audio_analysis_dialog import ArchiveAudioAnalysisDialog
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from ui.archive_audio_package_dialog import ArchiveAudioPackageDialog
from ui.custom_ui_widget.audio_data_manage_dialog import AudioDataManageDialog
from ui.segmented_analysis_results_dialog import SegmentedAnalysisResultsDialog
from unit_test.base.test_audio_record_filter import audio_row
from unit_test.ui.test_analysis_report_export_dialog import _index, FFT_IDENTITY, SPL_IDENTITY


@pytest.fixture(params=[(Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.KeypadModifier)],
                ids=["return", "keypad"])
def enter(request):
    return request.param


@pytest.fixture
def dialogs(ui_qapp):
    opened = []

    def show(dialog):
        if dialog not in opened:
            opened.append(dialog)
        dialog.show()
        dialog.activateWindow()
        ui_qapp.processEvents()
        return dialog

    yield show
    for dialog in reversed(opened):
        dialog.close()
        dialog.deleteLater()
    ui_qapp.processEvents()


def press(widget, enter, app):
    widget.setFocus()
    app.processEvents()
    focused = widget.window().focusWidget()
    assert focused is widget or (widget.focusProxy() is not None and focused is widget.focusProxy())
    QTest.keyClick(widget, *enter)
    app.processEvents()


def clicks(dialog):
    recorded = []
    for button in dialog.findChildren(QPushButton):
        button.clicked.connect(lambda checked=False, button=button: recorded.append(button))
    return recorded


@pytest.fixture
def report_io(monkeypatch):
    calls = SimpleNamespace(warnings=[], browse=Mock(), details=Mock(),
                            save=Mock(return_value=("", "")), thread=Mock())
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: calls.warnings.append(args[2]))
    monkeypatch.setattr(AnalysisReportExportDialog, "_choose_project", calls.browse)
    monkeypatch.setattr(AnalysisReportExportDialog, "_open_wav_details", calls.details)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", calls.save)
    monkeypatch.setattr("ui.analysis_report_export_dialog._ReportExportThread", calls.thread)
    return calls


@pytest.mark.parametrize("auxiliary", ["browse_button", "details_button", "cancel_button"])
def test_report_auxiliary_enter_reaches_existing_export(dialogs, report_io, tmp_path, enter, ui_qapp, auxiliary):
    dialog = dialogs(AnalysisReportExportDialog(auto_scan=False))
    dialog.load_index(_index(tmp_path))
    dialog.analysis_item_panel.set_checked(SPL_IDENTITY, True)
    recorded = clicks(dialog)
    press(getattr(dialog, auxiliary), enter, ui_qapp)
    assert recorded == [dialog.export_button]
    assert report_io.save.call_count == 1
    assert not report_io.warnings
    report_io.browse.assert_not_called()
    report_io.details.assert_not_called()
    report_io.thread.assert_not_called()  # Cancelled native save dialog.


@pytest.mark.parametrize("case,expected", [
    ("project", "请先选择项目目录。"),
    ("empty", "请选择型号。"),
    ("samples", "请至少选择一个样本编号。"),
    ("wav", "请至少勾选一个 WAV。"),
    ("items", "请至少勾选一个具体分析项。"),
])
def test_report_keyboard_keeps_validation(dialogs, report_io, tmp_path, enter, ui_qapp, case, expected):
    dialog = dialogs(AnalysisReportExportDialog(auto_scan=False))
    if case == "empty":
        dialog.load_index(ProjectReportIndex(str(tmp_path), "ProjectA", ()))
    elif case != "project":
        dialog.load_index(_index(tmp_path))
        if case == "samples":
            dialog.sample_selector.set_selected_values(())
        elif case == "wav":
            dialog.candidate_model.set_checked_paths(())
    recorded = clicks(dialog)
    press(dialog.browse_button, enter, ui_qapp)
    assert recorded == [dialog.export_button]
    assert report_io.warnings == [expected]
    report_io.save.assert_not_called()
    report_io.thread.assert_not_called()
    report_io.browse.assert_not_called()


def test_report_chart_only_enter_uses_chart_mode(dialogs, report_io, tmp_path, enter, ui_qapp):
    dialog = dialogs(AnalysisReportExportDialog(auto_scan=False))
    dialog.load_index(_index(tmp_path))
    dialog.analysis_item_panel.set_checked(FFT_IDENTITY, True)
    assert dialog._report_content() == "values_and_charts"
    assert not dialog.values_only_radio.isEnabled()
    press(dialog.values_and_charts_radio, enter, ui_qapp)
    assert report_io.save.call_count == 1
    assert not report_io.warnings
    report_io.thread.assert_not_called()


@pytest.mark.parametrize("editor", ["project_path_edit", "model_combo"])
def test_report_editors_do_not_export(dialogs, report_io, tmp_path, enter, ui_qapp, editor):
    dialog = dialogs(AnalysisReportExportDialog(auto_scan=False))
    dialog.load_index(_index(tmp_path))
    recorded = clicks(dialog)
    press(getattr(dialog, editor), enter, ui_qapp)
    assert recorded == []
    assert report_io.warnings == []
    report_io.save.assert_not_called()


@pytest.mark.parametrize("activity", ["scan", "export"])
def test_report_busy_enter_does_nothing(dialogs, report_io, enter, ui_qapp, activity):
    dialog = dialogs(AnalysisReportExportDialog(auto_scan=False))
    worker = SimpleNamespace(requestInterruption=Mock())
    setattr(dialog, f"_{activity}_thread", worker)
    dialog._set_busy(True)
    recorded = clicks(dialog)
    try:
        assert not dialog.export_button.isEnabled()
        press(dialog.cancel_button, enter, ui_qapp)
        assert recorded == []
        worker.requestInterruption.assert_not_called()
        report_io.save.assert_not_called()
        report_io.thread.assert_not_called()
    finally:
        setattr(dialog, f"_{activity}_thread", None)
        dialog._set_busy(False)


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("auxiliary", ["filter_button", "cancel_button"])
def test_wav_details_enter_confirms_even_empty(dialogs, monkeypatch, tmp_path, enter, ui_qapp, populated, auxiliary):
    opened_filter = Mock()
    monkeypatch.setattr(AnalysisReportWavDialog, "_show_filter_dialog", opened_filter)
    model = CandidateTableModel()
    model.reset_index(_index(tmp_path).candidates if populated else ())
    dialog = dialogs(AnalysisReportWavDialog(model))
    recorded = clicks(dialog)
    press(getattr(dialog, auxiliary), enter, ui_qapp)
    assert recorded == [dialog.confirm_button]
    assert dialog.result() == QDialog.Accepted
    opened_filter.assert_not_called()


def test_wav_filter_enter_applies_selection(dialogs, tmp_path, enter, ui_qapp):
    model = CandidateTableModel()
    model.reset_index(_index(tmp_path).candidates)
    details = dialogs(AnalysisReportWavDialog(model))
    dialog = dialogs(details.filter_dialog)
    dialog.reload()
    dialog.selector_by_column[CandidateTableModel.PORT_COLUMN].set_selected_values(("P2",))
    recorded = clicks(dialog)
    press(dialog.cancel_button, enter, ui_qapp)
    assert recorded == [dialog.apply_button]
    assert details.filter_model.rowCount() == 1
    assert dialog.result() == QDialog.Accepted


def test_archive_filter_protects_inputs_then_applies_without_reset(dialogs, enter, ui_qapp):
    rows = [audio_row("one")]
    dialog = dialogs(ArchiveAudioFilterDialog(rows, {r[0]: parse_audio_filter_metadata(r[1]) for r in rows}))
    recorded = clicks(dialog)
    dialog.date_filter_combobox.setEditText("2026-09-13")
    for widget in (dialog.combos["select_port"], dialog.date_filter_combobox.lineEdit()):
        press(widget, enter, ui_qapp)
        assert recorded == []
    press(dialog.reset_button, enter, ui_qapp)
    assert recorded == [dialog.apply_button]
    assert dialog.filter_config["select_record_date"] == "2026-09-13"
    assert dialog.result() == QDialog.Accepted


@pytest.mark.parametrize("selected", [False, True])
def test_package_enter_continues_only_with_selected_content(dialogs, enter, ui_qapp, selected):
    dialog = dialogs(ArchiveAudioPackageDialog(dict.fromkeys(("wav", "raw_csv", "images", "analysis_csv"), 1), 1))
    dialog.checkboxes["wav"].setChecked(selected)
    recorded = clicks(dialog)
    press(dialog.cancel_button, enter, ui_qapp)
    assert recorded == ([dialog.continue_button] if selected else [])
    assert dialog.result() == (QDialog.Accepted if selected else QDialog.Rejected)
    assert dialog.isVisible() is not selected


@pytest.fixture
def audio_services(monkeypatch):
    database = SimpleNamespace(get_record_audio_data=lambda: (error_code.OK, []),
                               query_stimulus_name_and_id=lambda: (error_code.OK, {}))
    monkeypatch.setattr("ui.custom_ui_widget.audio_data_manage_dialog.RecordingManager", lambda: database)
    playback = SimpleNamespace(is_audio_playing=lambda: False, stop_audio_playback=Mock())
    monkeypatch.setattr("ui.archive_audio_data_dialog.PlaybackController", lambda: playback)


def test_archive_management_enter_never_clicks_buttons(dialogs, audio_services, monkeypatch, enter, ui_qapp):
    actions = []
    for method in ("on_click_filter_btn", "show_all_wave", "on_clicked_order_btn", "on_clicked_package_btn", "on_clicked_delete_btn"):
        monkeypatch.setattr(ArchiveAudioDataDialog, method,
                            lambda self, checked=False, method=method: actions.append(method))
    dialog = dialogs(ArchiveAudioDataDialog(Mock()))
    recorded = clicks(dialog)
    buttons = dialog.findChildren(QPushButton)
    assert len(buttons) == 5
    for button in buttons:
        press(button, enter, ui_qapp)
    press(dialog.data_view, enter, ui_qapp)
    assert recorded == []
    assert actions == []
    QTest.mouseClick(dialog.order_btn, Qt.LeftButton)
    assert actions == ["on_clicked_order_btn"]


def test_audio_base_keeps_native_enter(dialogs, audio_services, monkeypatch, enter, ui_qapp):
    actions = []
    monkeypatch.setattr(AudioDataManageDialog, "on_click_filter_btn", lambda self: actions.append("filter"))
    archive = dialogs(ArchiveAudioDataDialog(Mock()))
    dialog = dialogs(AudioDataManageDialog(Mock()))
    assert not hasattr(dialog, "_enter_policy")
    button = dialog.top_layout.itemAt(dialog.top_layout.count() - 1).widget()
    press(button, enter, ui_qapp)
    assert actions == ["filter"]
    assert archive.isVisible()


def test_archive_result_enter_never_clicks_viewer_buttons(dialogs, monkeypatch, tmp_path, enter, ui_qapp):
    # No disk discovery or OS folder opening; retain real controls and key dispatch.
    monkeypatch.setattr(ArchiveAudioAnalysisDialog, "_discover", lambda self: None)
    monkeypatch.setattr(ArchiveAudioAnalysisDialog, "_open_folder", lambda self: None)
    dialog = dialogs(ArchiveAudioAnalysisDialog(tmp_path / "unused.wav"))
    recorded = clicks(dialog)
    for button in dialog.findChildren(QPushButton):
        button.show()
        button.setEnabled(True)
        press(button, enter, ui_qapp)
    assert recorded == []
    assert dialog.isVisible()


def test_segmented_results_explicitly_have_no_enter_target(dialogs, tmp_path, enter, ui_qapp):
    segment = AnalysisSegment(0, "时间1s", "time", 1, "s", 0, 100, 0, 100, 100)
    result = AnalysisTaskResult("task", "condition", str(tmp_path / "audio.wav"), "手动查看", "分析完成",
                                "未产生判定", None, (),
                                segments=(AnalysisSegmentResult(segment, (), "分析完成", "未产生判定", None),))
    dialog = dialogs(SegmentedAnalysisResultsDialog(result, {}))
    # A later viewer action must also remain protected by the explicit no-target policy.
    button = QPushButton("Viewer action", dialog)
    dialog.layout().addWidget(button)
    button.show()
    recorded = clicks(dialog)
    press(dialog.segment_selector, enter, ui_qapp)
    press(button, enter, ui_qapp)
    assert recorded == []
    assert dialog.isVisible()


def test_plain_result_widget_still_has_no_enter_action(dialogs, enter, ui_qapp):
    window = dialogs(AnalysisMultichannelResultWindow("SPL", ()))
    assert not window.findChildren(QPushButton)
    assert not hasattr(window, "_enter_policy")
    QTest.keyClick(window, *enter)
    assert window.isVisible()
