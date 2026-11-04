"""Caller-controlled station data and native read-only presentation."""

import math

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractItemView, QAbstractButton, QLabel, QScrollArea, QTableWidget, QVBoxLayout, QWidget

from ui.sequence.motor_station_test_panel import MotorStationTestPanel


@pytest.fixture
def panel(qt_app):
    widget = MotorStationTestPanel()
    widget.resize(340, 560)
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


@pytest.fixture
def styled_host(panel, qt_app):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    host.setStyleSheet("QWidget { background: #121212; color: #ffffff; } QLabel { color: #ff00ff; }")
    host.show()
    yield host
    panel.setParent(None)
    host.close()
    host.deleteLater()
    qt_app.processEvents()


def table(panel, station=1):
    return panel.findChild(QTableWidget, f"stationTable{station}")


def header(panel, station=1):
    return panel.findChild(QAbstractButton, f"stationHeader{station}")


def test_initial_zero_and_fixed_general_header(panel):
    assert panel.get_station_count() == 0
    assert not panel.findChildren(QTableWidget)
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert labels == ["工位测试", "工位", "正转", "反转"]
    assert len(panel.findChildren(QScrollArea)) == 1
    for name in ("stationTitle", "stationGeneralForward"):
        assert panel.findChild(QLabel, name).font().family() == "Microsoft YaHei UI"


def test_public_updates_and_independent_tables(panel):
    assert panel.set_station_count(2) is None
    assert panel.set_station_row_count(1, 2) is None
    panel.set_station_row_count(2, 1)
    assert panel.set_station_row(1, 0, ["SPL dB(A)", 68.5, 63.6]) is None
    assert panel.set_station_column(1, 2, [60, 61]) is None
    assert panel.set_station_cell(1, 1, 0, "响度 sone") is None
    assert panel.set_station_results(1, "ok", " NG ") is None
    snapshot = panel.get_station_data(1)
    assert snapshot == {"station_id": 1, "rows": [["SPL dB(A)", "68.5", "60"], ["响度 sone", "--", "61"]], "forward": "OK", "reverse": "NG"}
    assert panel.get_station_data(2)["rows"] == [["--", "--", "--"]]
    assert [[table(panel).item(r, c).text() for c in range(3)] for r in range(2)] == snapshot["rows"]
    assert table(panel).columnCount() == 3
    assert [table(panel).horizontalHeaderItem(i).text() for i in range(3)] == ["指标", "正转", "反转"]
    assert table(panel).editTriggers() == QAbstractItemView.NoEditTriggers
    assert table(panel).verticalHeader().isHidden()
    assert not table(panel).horizontalHeader().sectionsMovable()
    assert table(panel).item(0, 0).textAlignment() & Qt.AlignLeft
    assert table(panel).item(0, 1).textAlignment() & Qt.AlignRight


def test_independent_expansion_and_preserved_header(panel, qt_app):
    panel.set_station_count(2)
    panel.set_station_row_count(1, 2)
    header(panel).click()
    qt_app.processEvents()
    assert table(panel).isVisible()
    assert not table(panel, 2).isVisible()
    header(panel, 2).click()
    assert table(panel).isVisible() and table(panel, 2).isVisible()
    header(panel).click()
    assert not table(panel).isVisible() and table(panel, 2).isVisible()
    assert header(panel).isVisible()


def test_grow_shrink_same_counts_and_removed_data_reinitialize(panel):
    panel.set_station_count(2)
    panel.set_station_row_count(1, 2)
    panel.set_station_row(1, 0, ("kept", None, -3))
    panel.set_station_row(1, 1, ["removed", 0, 0])
    panel.set_station_results(1, "OK", "NG")
    header(panel).click()
    panel.set_station_count(2)
    panel.set_station_row_count(1, 2)
    assert header(panel).isChecked()
    panel.set_station_count(3)
    assert panel.get_station_data(3) == {"station_id": 3, "rows": [], "forward": "--", "reverse": "--"}
    panel.set_station_row_count(1, 1)
    panel.set_station_row_count(1, 3)
    assert panel.get_station_data(1)["rows"] == [["kept", "--", "-3"], ["--"] * 3, ["--"] * 3]
    assert header(panel).isChecked()
    panel.set_station_row_count(2, 1)
    panel.set_station_results(2, "NG", "OK")
    panel.set_station_count(1)
    panel.set_station_count(2)
    assert panel.get_station_data(2) == {"station_id": 2, "rows": [], "forward": "--", "reverse": "--"}
    assert not header(panel, 2).isChecked()
    panel.set_station_count(0)
    assert panel.get_station_count() == 0


def test_snapshot_and_caller_container_are_defensive(panel):
    panel.set_station_count(1)
    panel.set_station_row_count(1, 2)
    values = ["a", 1, 2.0]
    panel.set_station_row(1, 0, values)
    values[0] = "mutated"
    column = ["b", None]
    panel.set_station_column(1, 1, column)
    column[0] = "mutated"
    snapshot = panel.get_station_data(1)
    snapshot["rows"][0][0] = "mutated"
    snapshot["rows"].clear()
    snapshot["forward"] = "NG"
    assert panel.get_station_data(1) == {"station_id": 1, "rows": [["a", "b", "2.0"], ["--", "--", "--"]], "forward": "--", "reverse": "--"}


def test_instances_and_direction_states_are_independent(panel, qt_app):
    other = MotorStationTestPanel()
    try:
        panel.set_station_count(1)
        panel.set_station_row_count(1, 1)
        panel.set_station_row(1, 0, ["NG", "OK", "--"])
        assert panel.get_station_data(1)["forward"] == "--"
        panel.set_station_results(1, "OK", "NG")
        assert panel.get_station_data(1)["rows"] == [["NG", "OK", "--"]]
        assert other.get_station_count() == 0
    finally:
        other.close()
        other.deleteLater()
        qt_app.processEvents()


def test_zero_rows_accept_empty_columns(panel):
    panel.set_station_count(1)
    assert panel.set_station_column(1, 0, []) is None
    assert panel.set_station_column(1, 2, ()) is None
    assert panel.get_station_data(1)["rows"] == []


@pytest.mark.parametrize("method,args,error", [
    ("set_station_count", (True,), TypeError),
    ("set_station_count", (1.0,), TypeError),
    ("set_station_count", ("2",), TypeError),
    ("set_station_count", (-1,), ValueError),
    ("set_station_row_count", (1, False), TypeError),
    ("set_station_row_count", (1, None), TypeError),
    ("set_station_row_count", (1, -1), ValueError),
    ("set_station_row", (1, 0, "abc"), TypeError),
    ("set_station_row", (1, 0, iter([1, 2, 3])), TypeError),
    ("set_station_row", (1, 0, [1]), ValueError),
    ("set_station_row", (1, 0, [1, 2, 3, 4]), ValueError),
    ("set_station_column", (1, 1, {0: 1, 1: 2}), TypeError),
    ("set_station_column", (1, 1, "ab"), TypeError),
    ("set_station_column", (1, 1, [1]), ValueError),
    ("set_station_results", (1, "OK", "pending"), ValueError),
    ("set_station_results", (1, "OK", 3), TypeError),
    ("set_station_results", (1, None, "NG"), TypeError),
    ("set_station_results", (1, "", "NG"), ValueError),
])
def test_invalid_batches_leave_all_data_and_display_unchanged(panel, method, args, error):
    panel.set_station_count(2)
    panel.set_station_row_count(1, 2)
    panel.set_station_row(1, 0, ["original", 4, 5])
    header(panel).click()
    before = [panel.get_station_data(i) for i in (1, 2)]
    with pytest.raises(error, match=".+"):
        getattr(panel, method)(*args)
    assert [panel.get_station_data(i) for i in (1, 2)] == before
    assert [table(panel).item(0, c).text() for c in range(3)] == ["original", "4", "5"]
    assert header(panel).isChecked()


@pytest.mark.parametrize("bad,error", [(True, TypeError), ({}, TypeError), ([], TypeError), (object(), TypeError), (float("nan"), ValueError), (math.inf, ValueError), (-math.inf, ValueError)])
@pytest.mark.parametrize("method", ["set_station_cell", "set_station_row", "set_station_column"])
def test_invalid_scalar_data_is_atomic(panel, bad, error, method):
    panel.set_station_count(1)
    panel.set_station_row_count(1, 2)
    before = panel.get_station_data(1)
    args = {"set_station_cell": (1, 0, 1, bad), "set_station_row": (1, 0, ["valid", 1, bad]), "set_station_column": (1, 2, ["valid", bad])}[method]
    with pytest.raises(error, match=".+"):
        getattr(panel, method)(*args)
    assert panel.get_station_data(1) == before
    assert all(table(panel).item(r, c).text() == "--" for r in range(2) for c in range(3))


@pytest.mark.parametrize("station,error", [(True, TypeError), (1.0, TypeError), (None, TypeError), (0, IndexError), (-1, IndexError), (2, IndexError)])
@pytest.mark.parametrize("method,tail", [("set_station_row_count", (0,)), ("set_station_row", (0, [1, 2, 3])), ("set_station_column", (0, [1])), ("set_station_cell", (0, 0, 1)), ("set_station_results", ("OK", "NG")), ("get_station_data", ())])
def test_invalid_station_identifiers(panel, station, error, method, tail):
    panel.set_station_count(1)
    panel.set_station_row_count(1, 1)
    before = panel.get_station_data(1)
    with pytest.raises(error, match="station"):
        getattr(panel, method)(station, *tail)
    assert panel.get_station_data(1) == before


@pytest.mark.parametrize("index,error", [(True, TypeError), (0.0, TypeError), ("0", TypeError), (-1, IndexError), (3, IndexError)])
@pytest.mark.parametrize("method,target", [("set_station_row", "row"), ("set_station_cell", "row"), ("set_station_column", "column"), ("set_station_cell", "column")])
def test_invalid_table_indices(panel, index, error, method, target):
    panel.set_station_count(1)
    panel.set_station_row_count(1, 1)
    args = {("set_station_row", "row"): (1, index, [1, 2, 3]), ("set_station_cell", "row"): (1, index, 0, "value"), ("set_station_column", "column"): (1, index, [1]), ("set_station_cell", "column"): (1, 0, index, "value")}[method, target]
    with pytest.raises(error, match=target):
        getattr(panel, method)(*args)
    assert panel.get_station_data(1)["rows"] == [["--"] * 3]


@pytest.mark.parametrize("width", [340, 600])
def test_scrolling_alignment_and_native_status_colors(panel, qt_app, styled_host, width):
    # Simulate the application's generic inherited styles: station colors must win.
    styled_host.resize(width, 400)
    panel.set_station_count(9)
    panel.set_station_row_count(1, 2)
    panel.set_station_results(1, "OK", "NG")
    panel.set_station_row(1, 0, ["指标", 68.5, 63.6])
    header(panel).click()
    qt_app.processEvents()
    QTest.qWait(10)  # Qt posts another layout pass after content minimum-size changes.
    scroll = panel.findChild(QScrollArea)
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    assert not scroll.horizontalScrollBar().isVisible()
    assert table(panel).horizontalScrollBar().maximum() == 0
    assert table(panel).width() <= scroll.viewport().width()
    detail_header = table(panel).horizontalHeader()
    assert detail_header.grab().toImage().pixelColor(3, 3) == QColor("#f1f5f9")
    general_header = panel.findChild(QWidget, "stationGeneralHeader")
    assert general_header.grab().toImage().pixelColor(3, 3) == QColor("#f1f5f9")
    general = panel.findChild(QLabel, "stationGeneralForward")
    forward = panel.findChild(QLabel, "stationForward1")
    reverse = panel.findChild(QLabel, "stationReverse1")
    pending = panel.findChild(QLabel, "stationForward2")
    for label, color in [(forward, "#00824e"), (reverse, "#ca3245"), (pending, "#617183")]:
        assert label.palette().color(label.foregroundRole()) == QColor(color)
    assert abs(general.mapTo(panel, general.rect().center()).x() - forward.mapTo(panel, forward.rect().center()).x()) <= 1
    assert abs(table(panel).columnWidth(0) / table(panel).viewport().width() - .46) < .02
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    assert general.isVisible()


@pytest.mark.parametrize("count", [0, 2, 18])
def test_header_background_fills_inner_width_through_resize(panel, qt_app, styled_host, count):
    panel.set_station_count(count)
    if count:
        panel.set_station_row_count(1, 2)
        header(panel).click()
    for width in (600, 340, 600, 340):
        styled_host.resize(width, 400)
        qt_app.processEvents()
        QTest.qWait(10)
        qt_app.processEvents()
        title = panel.findChild(QLabel, "stationTitle")
        general = panel.findChild(QWidget, "stationGeneralHeader")
        scroll = panel.findChild(QScrollArea)
        image = panel.grab().toImage()
        assert title.geometry().right() == panel.width() - 2
        assert general.geometry().right() == panel.width() - 2
        assert image.pixelColor(panel.width() - 2, title.geometry().center().y()) == QColor("#3e75ac")
        assert image.pixelColor(panel.width() - 2, general.y() + 4) == QColor("#f1f5f9")
        assert image.pixelColor(panel.width() - 1, panel.height() // 2) == QColor("#cddaea")
        assert scroll.horizontalScrollBar().maximum() == 0
        if count:
            for name in ("Forward", "Reverse"):
                label = panel.findChild(QLabel, "stationGeneral" + name)
                station_label = panel.findChild(QLabel, "station" + name + "1")
                assert abs(label.mapTo(panel, label.rect().center()).x() -
                           station_label.mapTo(panel, station_label.rect().center()).x()) <= 1
    # Removing the scrollbar must restore the full column width too.
    panel.set_station_count(0)
    qt_app.processEvents()
    QTest.qWait(10)
    assert panel._general_header.contentsRect().width() == panel._scroll.viewport().width()
