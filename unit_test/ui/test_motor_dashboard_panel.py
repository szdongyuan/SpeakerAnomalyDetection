"""Template summary and independently scrollable station composition."""

from datetime import date
import builtins
import sqlite3

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ui.sequence.motor_information_summary_panel import MotorInformationSummaryPanel


@pytest.fixture
def summary(qt_app):
    widget = MotorInformationSummaryPanel()
    widget.resize(340, widget.sizeHint().height())
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def test_summary_shows_local_date_and_initial_template(summary):
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
    assert not button.isEnabled()


def test_summary_has_no_database_history_or_countboard_dependency(qt_app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Summary template must not access history, databases or count board")

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
        assert not widget.findChildren(QDialog)
        assert not any(isinstance(top, QDialog) and top.isVisible() for top in qt_app.topLevelWidgets())
    finally:
        widget.close()
        widget.deleteLater()
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


@pytest.mark.parametrize("width", [340, 600])
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
    assert dashboard.rect().contains(summary_rect)
    for child in dashboard.summary_panel.findChildren(QLabel) + dashboard.summary_panel.findChildren(QPushButton):
        assert child.isVisibleTo(dashboard)
        assert child.visibleRegion().boundingRect() == child.rect()
    dashboard.resize(width, 700)
    qt_app.processEvents()
    assert dashboard.summary_panel.height() == summary_rect.height()
    assert station.height() > summary_rect.top()


def test_dashboard_instances_keep_station_data_and_template_independent(dashboard, qt_app):
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
    dashboard.resize(340, 400)
    qt_app.processEvents()
    QTest.qWait(10)
    assert dashboard.width() == 340
    assert dashboard.minimumSizeHint().width() <= 340
    assert station.minimumSizeHint().width() <= 340
    scroll = station.findChild(QScrollArea)
    general_header = station.findChild(QWidget, "stationGeneralHeader")
    assert general_header.width() == scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() == 0
    assert dashboard.rect().contains(dashboard.summary_panel.geometry())
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
        for name, color in [("summaryOk", "#00824e"), ("summaryNg", "#ca3245"), ("summaryUnmarked", "#617183")]:
            label = summary.findChild(QLabel, name)
            expected = QColor(color)
            assert label.palette().color(label.foregroundRole()) == expected
            pixels = label.grab().toImage()
            assert any(pixels.pixelColor(x, y) == expected for x in range(pixels.width()) for y in range(pixels.height()))
            assert label.font().family() == "Microsoft YaHei UI"
            assert label.font().pointSizeF() == 17.25
        title = summary.findChild(QLabel, "summaryTitle")
        assert title.font().pointSizeF() == 10.5
        button = summary.findChild(QPushButton, "summaryDetails")
        assert button.palette().color(QPalette.Disabled, QPalette.ButtonText) == QColor("#246bed")
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
