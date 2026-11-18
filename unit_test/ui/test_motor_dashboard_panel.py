"""Caller-owned daily summary and independently scrollable station composition."""

from datetime import date
import builtins
import sqlite3

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from base.workbench_summary_store import WorkbenchSummarySnapshot
from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY_NAME
from ui.sequence.motor_information_summary_panel import MotorInformationSummaryPanel
from unit_test.daily_record_cases import make_daily_records, make_daily_wav_records


@pytest.fixture
def summary(qt_app):
    widget = MotorInformationSummaryPanel()
    widget.resize(340, widget.sizeHint().height())
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()


def test_summary_shows_local_date_and_initial_empty_snapshot(summary):
    expected = {
        "summaryTitle": "信息汇总", "summaryDate": date.today().isoformat(),
        "summaryTotalCaption": "总数", "summaryTotal": "0",
        "summaryRateCaption": "合格率", "summaryRate": "--",
        "summaryOkCaption": "OK", "summaryOk": "0",
        "summaryNgCaption": "NG", "summaryNg": "0",
        "summaryUnmarkedCaption": "未标记", "summaryUnmarked": "0",
    }
    assert {label.objectName(): label.text() for label in summary.findChildren(QLabel)} == expected
    button = summary.findChild(QPushButton, "summaryDetails")
    assert button.text() == "查看明细"
    assert button.isEnabled()
    assert summary._details_dialog is None


def test_summary_has_no_database_history_or_countboard_dependency(qt_app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Daily summary must not access history, databases or count board")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("base.db_manager", "ui.sequence.sequencement_count_board") or "history" in name:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(QDialog, "exec_", forbidden)
    widget = MotorInformationSummaryPanel()
    try:
        widget.show()
        button = widget.findChild(QPushButton, "summaryDetails")
        button.click()
        QTest.mouseClick(button, Qt.LeftButton)
        qt_app.processEvents()
        dialog = widget._details_dialog
        assert isinstance(dialog, QDialog)
        assert dialog.isWindow()
        assert dialog.windowModality() == Qt.ApplicationModal
        assert qt_app.activeModalWidget() is dialog
        assert [top for top in qt_app.topLevelWidgets()
                if isinstance(top, QDialog) and top.isVisible()] == [dialog]
    finally:
        for dialog in widget.findChildren(QDialog):
            dialog.close()
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()


@pytest.fixture
def dashboard(qt_app):
    from ui.sequence.motor_dashboard_panel import MotorDashboardPanel

    widget = MotorDashboardPanel()
    widget.resize(340, 560)
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def test_dashboard_owns_station_area_above_summary_with_zero_stations(dashboard):
    from ui.sequence.motor_station_test_panel import MotorStationTestPanel

    assert isinstance(dashboard.station_panel, MotorStationTestPanel)
    assert isinstance(dashboard.summary_panel, MotorInformationSummaryPanel)
    assert dashboard.station_panel.parentWidget() is dashboard
    assert dashboard.summary_panel.parentWidget() is dashboard
    assert dashboard.station_panel.get_station_count() == 0
    assert dashboard.station_panel.geometry().bottom() < dashboard.summary_panel.y()
    assert dashboard.summary_panel.isVisible()
    assert len(dashboard.findChildren(QScrollArea)) == 1


def assert_summary_fits_dashboard(dashboard):
    summary = dashboard.summary_panel
    assert summary.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert summary.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
    assert summary.minimumWidth() == 0
    assert summary.maximumWidth() == 16777215
    assert summary.width() == dashboard.width()
    assert dashboard.rect().contains(summary.geometry())
    children = summary.findChildren(QLabel) + summary.findChildren(QPushButton)
    rectangles = []
    for child in children:
        assert child.isVisibleTo(dashboard)
        assert child.visibleRegion().boundingRect() == child.rect()
        assert child.width() >= child.sizeHint().width()
        assert child.height() >= child.sizeHint().height()
        rectangle = child.rect().translated(child.mapTo(summary, QPoint(0, 0)))
        assert summary.rect().contains(rectangle)
        assert all(not rectangle.intersects(previous) for previous in rectangles)
        rectangles.append(rectangle)
    counts = summary.findChild(QWidget, "summaryCounts")
    columns = counts.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
    assert len(columns) == 3
    assert max(column.width() for column in columns) - min(column.width() for column in columns) <= 1


@pytest.mark.parametrize("width", [300, 340, 600])
def test_summary_stays_visible_when_station_content_scrolls_and_resizes(dashboard, qt_app, width):
    dashboard.resize(width, 400)
    station = dashboard.station_panel
    station.set_station_count(18)
    station.set_station_row_count(1, 8)
    station.findChild(QPushButton, "stationHeader1").click()
    qt_app.processEvents()
    QTest.qWait(10)
    scroll = station.findChild(QScrollArea)
    summary_rect = dashboard.summary_panel.geometry()
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qt_app.processEvents()
    assert dashboard.summary_panel.geometry() == summary_rect
    assert_summary_fits_dashboard(dashboard)
    dashboard.resize(width, 700)
    qt_app.processEvents()
    assert dashboard.summary_panel.height() == summary_rect.height()
    assert station.height() > summary_rect.top()
    assert_summary_fits_dashboard(dashboard)


def test_dashboard_instances_keep_station_data_and_daily_summary_independent(dashboard, qt_app):
    from ui.sequence.motor_dashboard_panel import MotorDashboardPanel

    other = MotorDashboardPanel()
    try:
        dashboard.station_panel.set_station_count(1)
        dashboard.station_panel.set_station_row_count(1, 1)
        dashboard.station_panel.set_station_row(1, 0, ["SPL", 68.5, 63.6])
        dashboard.station_panel.set_station_results(1, "OK", "NG")
        assert other.station_panel.get_station_count() == 0
        assert dashboard.summary_panel is not other.summary_panel
        for widget in (dashboard.summary_panel, other.summary_panel):
            assert widget.findChild(QLabel, "summaryTotal").text() == "0"
            assert widget.findChild(QLabel, "summaryRate").text() == "--"
            assert [widget.findChild(QLabel, f"summary{name}").text() for name in ("Ok", "Ng", "Unmarked")] == ["0"] * 3
    finally:
        other.close()
        other.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize("station_count", [0, 18])
def test_dashboard_can_shrink_after_widening_without_header_minimum_width(dashboard, qt_app, station_count):
    station = dashboard.station_panel
    station.set_station_count(station_count)
    if station_count:
        station.set_station_row_count(1, 2)
        station.findChild(QPushButton, "stationHeader1").click()
    dashboard.resize(600, 400)
    qt_app.processEvents()
    QTest.qWait(10)
    assert dashboard.width() == 600
    dashboard.resize(300, 400)
    qt_app.processEvents()
    QTest.qWait(10)
    assert dashboard.width() == 300
    assert dashboard.minimumSizeHint().width() <= 300
    assert station.minimumSizeHint().width() <= 300
    scroll = station.findChild(QScrollArea)
    general_header = station.findChild(QWidget, "stationGeneralHeader")
    assert general_header.contentsRect().width() == scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() == 0
    assert_summary_fits_dashboard(dashboard)
    if station_count:
        assert scroll.verticalScrollBar().maximum() > 0
        general = station.findChild(QLabel, "stationGeneralForward")
        forward = station.findChild(QLabel, "stationForward1")
        assert abs(general.mapTo(station, general.rect().center()).x() - forward.mapTo(station, forward.rect().center()).x()) <= 1


def test_summary_native_render_matches_html_under_inherited_styles(summary, qt_app):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(summary)
    host.setStyleSheet("QWidget { background: #121212; color: #ffffff; } QLabel { color: #ff00ff; }")
    host.resize(340, summary.sizeHint().height())
    host.show()
    qt_app.processEvents()
    try:
        assert summary.grab().toImage().pixelColor(5, 5) == QColor("#ffffff")
        for name, color in [("summaryOk", "#009900"), ("summaryNg", "#ca3245"), ("summaryUnmarked", "#617183")]:
            label = summary.findChild(QLabel, name)
            expected = QColor(color)
            assert label.palette().color(label.foregroundRole()) == expected
            pixels = label.grab().toImage()
            assert any(pixels.pixelColor(x, y) == expected for x in range(pixels.width()) for y in range(pixels.height()))
            assert label.font().family() == MAIN_UI_SMALL_FONT_FAMILY_NAME
            assert label.font().pixelSize() == 26
            assert label.font().bold()
        title = summary.findChild(QLabel, "summaryTitle")
        assert title.font().pixelSize() == 17
        assert title.font().bold()
        for name, size in (("summaryDate", 14), ("summaryTotal", 18), ("summaryRate", 18)):
            font = summary.findChild(QLabel, name).font()
            assert (font.family(), font.pixelSize()) == (MAIN_UI_SMALL_FONT_FAMILY_NAME, size)
            assert font.bold() == (name != "summaryDate")
        for name in ("Total", "Rate", "Ok", "Ng", "Unmarked"):
            label = summary.findChild(QLabel, f"summary{name}Caption")
            assert (label.font().family(), label.font().pixelSize()) == (MAIN_UI_SMALL_FONT_FAMILY_NAME, 15)
            assert not label.font().bold()
            assert label.palette().color(label.foregroundRole()) == QColor("#617183")
        button = summary.findChild(QPushButton, "summaryDetails")
        assert (button.font().family(), button.font().pixelSize()) == (MAIN_UI_SMALL_FONT_FAMILY_NAME, 15)
        assert not button.font().bold()
        assert button.minimumHeight() == 24
        expected = QColor("#246bed")
        assert button.isEnabled()
        assert button.palette().color(QPalette.Active, QPalette.ButtonText) == expected
        pixels = button.grab().toImage()
        assert any(pixels.pixelColor(x, y) == expected
                   for x in range(pixels.width()) for y in range(pixels.height()))
        counts = summary.findChild(QWidget, "summaryCounts")
        image = counts.grab().toImage()
        assert image.pixelColor(5, 0) == QColor("#e0e7ef")
        columns = counts.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
        assert len(columns) == 3
        assert max(column.width() for column in columns) - min(column.width() for column in columns) <= 1
        for column in columns[1:]:
            point = column.mapTo(counts, QPoint(0, column.height() // 2))
            assert image.pixelColor(point) == QColor("#e0e7ef")
    finally:
        summary.setParent(None)
        host.close()
        host.deleteLater()
        qt_app.processEvents()


def summary_values(summary):
    return tuple(summary.findChild(QLabel, f"summary{name}").text()
                 for name in ("Date", "Total", "Ok", "Ng", "Unmarked", "Rate"))


def dialog_values(dialog):
    return tuple(getattr(dialog, f"{name}_label").text()
                 for name in ("date", "total", "ok", "ng", "unmarked", "rate"))


def test_count_snapshot_updates_home_and_preserves_open_wav_details(qt_app):
    from ui.sequence.motor_left_panel import MotorDetectionLeftPanel

    left = MotorDetectionLeftPanel()
    try:
        wav_day = date(2026, 11, 4)
        left.set_daily_wav_records(make_daily_wav_records(123, wav_day), wav_day)
        summary = left.dashboard_panel.summary_panel
        summary.findChild(QPushButton, "summaryDetails").click()
        dialog = summary._details_dialog
        dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(2))
        dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("NG"))
        dialog.search_edit.setText(" case ")
        dialog.next_button.click()
        details = dialog_values(dialog)
        rows = [dialog.table.item(row, 2).text() for row in range(dialog.table.rowCount())]
        assert dialog.page_label.text() == "2 / 3"

        home_day = date(2026, 11, 5)
        assert left.set_summary_counts(WorkbenchSummarySnapshot(home_day, 2, 1, 1)) is None
        assert summary_values(summary) == (home_day.isoformat(), "4", "2", "1", "1", "50.0%")
        assert summary._details_dialog is dialog and dialog.isVisible()
        assert dialog_values(dialog) == details
        assert [dialog.table.item(row, 2).text() for row in range(dialog.table.rowCount())] == rows
        assert dialog.page_label.text() == "2 / 3"
        assert dialog.station_combo.currentData() == 2
        assert dialog.result_combo.currentData() == "NG"
        assert dialog.search_edit.text() == " case "
    finally:
        left.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()


def test_round_and_wav_snapshots_have_independent_statistics_and_dates(summary, qt_app):
    round_day = date(2026, 11, 4)
    wav_day = date(2026, 11, 5)
    rounds = make_daily_records(1, round_day)
    rounds[0]["result"] = "NG"
    wavs = make_daily_wav_records(2, wav_day)
    assert summary.set_daily_records(rounds, round_day) is None
    assert summary.set_daily_wav_records(wavs, wav_day) is None
    round_expected = (round_day.isoformat(), "1", "0", "1", "0", "0.0%")
    wav_expected = (wav_day.isoformat(), "2", "1", "1", "0", "50.0%")
    assert summary_values(summary) == round_expected
    assert summary._details_dialog is None
    rounds.clear()
    wavs[0]["result"] = "NG"
    wavs.clear()
    summary.findChild(QPushButton, "summaryDetails").click()
    qt_app.processEvents()
    dialog = summary._details_dialog
    assert dialog_values(dialog) == wav_expected
    assert dialog.table.rowCount() == 2
    assert [dialog.table.item(i, 4).text() for i in range(2)] == ["NG", "OK"]
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("OK"))
    dialog.search_edit.setText("case")
    assert dialog.table.rowCount() == 1
    assert summary_values(summary) == round_expected
    assert dialog_values(dialog) == wav_expected
    clear_day = date(2026, 11, 6)
    assert summary.set_daily_wav_records([], clear_day) is None
    assert summary_values(summary) == round_expected
    assert dialog_values(dialog) == (clear_day.isoformat(), "0", "0", "0", "0", "--")
    assert dialog.table.rowCount() == 0
    assert dialog.result_combo.currentData() == "OK"
    assert dialog.search_edit.text() == "case"
    dialog.close()


def test_round_snapshot_alone_does_not_create_wav_details(summary):
    day = date(2026, 11, 4)
    summary.set_daily_records(make_daily_records(41, day), day)
    summary.findChild(QPushButton, "summaryDetails").click()
    dialog = summary._details_dialog
    assert summary_values(summary) == (day.isoformat(), "41", "14", "14", "13", "34.1%")
    assert dialog_values(dialog) == (date.today().isoformat(), "0", "0", "0", "0", "--")
    assert dialog.table.rowCount() == 0


def test_round_replacement_and_clear_leave_wav_filters_page_date_and_statistics_intact(summary):
    day = date(2026, 11, 4)
    summary.set_daily_wav_records(make_daily_wav_records(123, day), day)
    summary.findChild(QPushButton, "summaryDetails").click()
    dialog = summary._details_dialog
    dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(2))
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("NG"))
    dialog.search_edit.setText(" case ")
    dialog.next_button.click()
    assert dialog.page_label.text() == "2 / 3"
    before = (dialog_values(dialog), dialog.page_label.text(), dialog.table.rowCount(),
              dialog.table.item(0, 2).text(), dialog.station_combo.currentData(),
              dialog.result_combo.currentData(), dialog.search_edit.text())
    other_day = date(2026, 11, 6)
    for records in (make_daily_records(1, other_day), []):
        summary.set_daily_records(records, other_day)
        assert (dialog_values(dialog), dialog.page_label.text(), dialog.table.rowCount(),
                dialog.table.item(0, 2).text(), dialog.station_combo.currentData(),
                dialog.result_combo.currentData(), dialog.search_edit.text()) == before
    assert summary_values(summary) == (other_day.isoformat(), "0", "0", "0", "0", "--")


@pytest.mark.parametrize("close_method", ["escape", "system"])
def test_daily_dialog_reuses_instance_and_preserves_filters_and_page(summary, qt_app, close_method):
    day = date(2026, 11, 4)
    summary.set_daily_wav_records(make_daily_wav_records(123, day), day)
    button = summary.findChild(QPushButton, "summaryDetails")
    button.click()
    dialog = summary._details_dialog
    dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(2))
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("NG"))
    dialog.search_edit.setText(" case ")
    dialog.next_button.click()
    assert dialog.page_label.text() == "2 / 3"
    button.click()
    assert summary._details_dialog is dialog
    assert len(summary.findChildren(QDialog)) == 1
    if close_method == "escape":
        QTest.keyClick(dialog, Qt.Key_Escape)
    else:
        dialog.close()
    qt_app.processEvents()
    assert not dialog.isVisible()
    assert qt_app.activeModalWidget() is None
    button.click()
    qt_app.processEvents()
    assert summary._details_dialog is dialog
    assert qt_app.activeModalWidget() is dialog
    assert dialog.station_combo.currentData() == 2
    assert dialog.result_combo.currentData() == "NG"
    assert dialog.search_edit.text() == " case "
    assert dialog.page_label.text() == "2 / 3"
    dialog.close()


@pytest.mark.parametrize("snapshot_kind", ["round", "wav"])
@pytest.mark.parametrize("invalid_kind", ["result", "station", "container", "date"])
def test_invalid_daily_replacement_preserves_snapshot_ui_and_window_state(summary, snapshot_kind, invalid_kind):
    day = date(2026, 11, 4)
    summary.set_daily_records(make_daily_records(3, day), day)
    summary.set_daily_wav_records(make_daily_wav_records(123, day), day)
    summary.findChild(QPushButton, "summaryDetails").click()
    dialog = summary._details_dialog
    dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(2))
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("NG"))
    dialog.search_edit.setText("case")
    dialog.next_button.click()
    before = (summary_values(summary), dialog_values(dialog), dialog.page_label.text(),
              dialog.table.rowCount(), dialog.table.item(0, 2).text(),
              summary._daily_records.records, summary._daily_wav_records.records)
    new_day = date(2026, 11, 5)
    make_records = make_daily_records if snapshot_kind == "round" else make_daily_wav_records
    publish = summary.set_daily_records if snapshot_kind == "round" else summary.set_daily_wav_records
    invalid = make_records(2, new_day)
    if invalid_kind == "result":
        invalid[-1]["result"] = "invalid"
        error, field = ValueError, "result"
    elif invalid_kind == "station":
        invalid[-1]["station_id"] = False
        error, field = TypeError, "station_id"
    elif invalid_kind == "container":
        invalid = iter(invalid)
        error, field = TypeError, "records"
    else:
        new_day = invalid[0]["completed_at"]
        error, field = TypeError, "record_date"
    with pytest.raises(error, match=field):
        publish(invalid, new_day)
    assert (summary_values(summary), dialog_values(dialog), dialog.page_label.text(),
            dialog.table.rowCount(), dialog.table.item(0, 2).text(),
            summary._daily_records.records, summary._daily_wav_records.records) == before
    assert dialog.station_combo.currentData() == 2
    assert dialog.result_combo.currentData() == "NG"
    assert dialog.search_edit.text() == "case"
    assert summary._details_dialog is dialog
    dialog.close()
    new_day = date(2026, 11, 5)
    summary.set_daily_wav_records(make_daily_wav_records(63, new_day), new_day)
    assert dialog.page_label.text() == "1 / 2"
    assert dialog.station_combo.currentData() == 2
    assert dialog.result_combo.currentData() == "NG"
    assert dialog.search_edit.text() == "case"
    assert dialog.date_label.text() == new_day.isoformat()
    assert summary_values(summary) == before[0]


def test_daily_summary_instances_keep_data_and_dialog_conditions_independent(summary, qt_app):
    other = MotorInformationSummaryPanel()
    try:
        day = date(2026, 11, 4)
        summary.set_daily_records(make_daily_records(21, day), day)
        summary.set_daily_wav_records(make_daily_wav_records(21, day), day)
        other.set_daily_records(make_daily_records(1, day), day)
        other.set_daily_wav_records(make_daily_wav_records(1, day), day)
        summary.findChild(QPushButton, "summaryDetails").click()
        first = summary._details_dialog
        first.next_button.click()
        first.close()
        other.findChild(QPushButton, "summaryDetails").click()
        second = other._details_dialog
        assert first is not second
        assert first.page_label.text() == "2 / 2"
        assert second.page_label.text() == "1 / 1"
        second.search_edit.setText("no match")
        assert first.search_edit.text() == ""
        other.set_daily_records([], day)
        other.set_daily_wav_records([], day)
        assert summary_values(summary)[1] == "21"
        assert first.table.rowCount() == 1
        second.close()
    finally:
        other.close()
        other.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()


def test_destroying_summary_releases_dialog_owned_by_separate_top_level(qt_app):
    from PyQt5 import sip

    host = QWidget()
    widget = MotorInformationSummaryPanel(host)
    host.show()
    widget.findChild(QPushButton, "summaryDetails").click()
    dialog = widget._details_dialog
    try:
        assert dialog.parentWidget() is host
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()
        assert sip.isdeleted(widget)
        assert sip.isdeleted(dialog)
        assert not sip.isdeleted(host)
        assert qt_app.activeModalWidget() is None
    finally:
        host.close()
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_destroying_previous_host_clears_deleted_dialog_wrapper_for_surviving_summary(qt_app):
    from PyQt5 import sip

    host = QWidget()
    new_host = QWidget()
    widget = MotorInformationSummaryPanel(host)
    widget.findChild(QPushButton, "summaryDetails").click()
    old_dialog = widget._details_dialog
    try:
        widget.setParent(new_host)
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()
        assert sip.isdeleted(old_dialog)
        assert widget._details_dialog is None
        widget.set_daily_records([], date(2026, 11, 4))
        widget.findChild(QPushButton, "summaryDetails").click()
        assert widget._details_dialog is not old_dialog
        assert widget._details_dialog.parentWidget() is new_host
        widget._details_dialog.close()
    finally:
        if not sip.isdeleted(host):
            host.deleteLater()
        new_host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()


def test_left_panel_forwards_independent_wav_snapshot(qt_app):
    from ui.sequence.motor_left_panel import MotorDetectionLeftPanel

    left = MotorDetectionLeftPanel()
    try:
        day = date(2026, 11, 4)
        assert left.set_daily_records(make_daily_records(1, day), day) is None
        assert left.set_daily_wav_records(make_daily_wav_records(2, day), day) is None
        summary = left.dashboard_panel.summary_panel
        summary.findChild(QPushButton, "summaryDetails").click()
        assert summary_values(summary) == (day.isoformat(), "1", "1", "0", "0", "100.0%")
        assert dialog_values(summary._details_dialog) == (day.isoformat(), "2", "1", "1", "0", "50.0%")
        summary._details_dialog.close()
    finally:
        left.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qt_app.processEvents()
