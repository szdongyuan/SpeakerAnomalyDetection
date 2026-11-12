"""Caller-controlled station data and native read-only presentation."""

import math

import pytest
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAbstractButton, QAbstractItemView, QLabel, QScrollArea, QStyle, QStyleOption,
    QTableWidget, QVBoxLayout, QWidget,
)

from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY_NAME
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
    assert labels == ["工位测试", "工位", "正转", "反转", "结果"]
    assert len(panel.findChildren(QScrollArea)) == 1
    for name, size, bold in (("stationTitle", 14, True), ("stationGeneralForward", 12, False), ("stationGeneralResult", 12, False)):
        label = panel.findChild(QLabel, name)
        label.ensurePolished()
        font = label.font()
        assert (font.family(), font.pixelSize(), font.bold()) == (MAIN_UI_SMALL_FONT_FAMILY_NAME, size, bold)


def test_station_details_use_pixel_fonts(panel):
    panel.set_station_count(1)
    for widget in (panel.findChild(QLabel, "stationForward1"), panel.findChild(QLabel, "stationResult1"), table(panel)):
        widget.ensurePolished()
        font = widget.font()
        assert (font.family(), font.pixelSize(), font.bold()) == (MAIN_UI_SMALL_FONT_FAMILY_NAME, 13, False)


def test_workbench_stations_show_names_and_reset_results_on_same_size_update(panel, qt_app):
    panel.set_station_count(1)
    assert header(panel).accessibleName() == "工位 1"
    assert panel.get_station_data(1)["forward"] == "--"
    assert panel.get_station_data(1)["reverse"] == "--"

    panel.show_workbench_stations((("左侧工位", ("响度",)), ("右侧工位", ())))
    qt_app.processEvents()
    assert panel.get_station_count() == 2
    first_header = header(panel)
    assert first_header.accessibleName() == "左侧工位"
    name_label = first_header.findChildren(QLabel)[0]
    assert name_label.text() == "左侧工位"
    assert not first_header.isChecked() and table(panel).isHidden()
    assert first_header.accessibleDescription() == "收起"
    first_header.click()
    assert first_header.isChecked() and table(panel).isVisible()
    assert first_header.accessibleDescription() == "展开"
    assert name_label.text() == "左侧工位"
    first_header.click()
    assert not first_header.isChecked() and table(panel).isHidden()
    assert first_header.accessibleDescription() == "收起"
    first_header.click()
    assert header(panel, 2).accessibleName() == "右侧工位"
    assert panel.get_station_data(1)["rows"] == [["响度", "--", "--"]]
    assert panel.get_station_data(2)["rows"] == []
    panel.set_station_row(1, 0, ("响度", 10, 20))
    panel.set_station_results(1, "OK", "NG")
    panel.set_station_results(2, "NG", "OK")

    panel.show_workbench_stations((("电机工位", ("振动",)), ("备件工位", ())))
    assert header(panel).accessibleName() == "电机工位"
    assert name_label.text() == "电机工位"
    assert header(panel).isChecked() and table(panel).isVisible()
    assert not header(panel, 2).isChecked() and table(panel, 2).isHidden()
    assert panel.get_station_data(1) == {
        "station_id": 1, "forward": "--", "reverse": "--", "result": "--", "rows": [["振动", "--", "--"]],
    }
    assert panel.get_station_data(2) == {
        "station_id": 2, "forward": "--", "reverse": "--", "result": "--", "rows": [],
    }
    assert [table(panel).item(0, c).text() for c in range(3)] == ["振动", "--", "--"]
    for name in ("stationForward1", "stationReverse1", "stationForward2", "stationReverse2"):
        assert panel.findChild(QLabel, name).text() == "--"

    panel.show_workbench_stations(())
    assert panel.get_station_count() == 0
    assert not panel.findChildren(QTableWidget)
    assert panel.findChild(QLabel, "stationTitle").isVisible()


def test_workbench_stations_preserve_empty_and_whitespace_metric_names(panel):
    panel.show_workbench_stations((("工位", ("", " ")),))

    rows = [["", "--", "--"], [" ", "--", "--"]]
    assert panel.get_station_data(1)["rows"] == rows
    assert [[table(panel).item(r, c).text() for c in range(3)] for r in range(2)] == rows


def test_workbench_station_header_shows_markup_as_literal_text(panel):
    name = "<h1>motor</h1>"
    panel.show_workbench_stations(((name, ()),))

    station_header = header(panel)
    label = station_header.findChildren(QLabel)[0]
    assert label.text() == name
    assert label.textFormat() == Qt.PlainText
    assert station_header.accessibleName() == name
    station_header.click()
    assert label.text() == name


def native_branch_mask(style, button, slot, expanded, image):
    reference = QImage(image.size(), QImage.Format_ARGB32_Premultiplied)
    reference.setDevicePixelRatio(image.devicePixelRatio())
    reference.fill(Qt.transparent)
    option = QStyleOption()
    option.initFrom(button)
    option.rect = QRect(QPoint(), slot.size())
    option.state |= QStyle.State_Children
    if expanded:
        option.state |= QStyle.State_Open
    with QPainter(reference) as painter:
        style.drawPrimitive(QStyle.PE_IndicatorBranch, option, painter, button)
    return {(x, y) for y in range(reference.height()) for x in range(reference.width())
            if reference.pixelColor(x, y).alpha()}


def test_native_branch_arrows_follow_only_the_operated_station(panel, qt_app):
    panel.show_workbench_stations((("左侧工位", ("响度",)), ("右侧工位", ("响度",))))
    first, second = header(panel), header(panel, 2)
    second.setChecked(True)
    qt_app.processEvents()

    style = qt_app.style()
    slot_width = style.pixelMetric(QStyle.PM_TreeViewIndentation)
    slot = QRect(12, 0, slot_width, first.height())
    name = first.findChildren(QLabel)[0]
    text_left = name.x() + name.contentsRect().left()

    def assert_branch(button, expanded):
        image = button.grab(slot).toImage()
        background = button.grab(QRect(0, 0, 1, slot.height())).toImage()
        mask = {(x, y) for y in range(image.height()) for x in range(image.width())
                if image.pixel(x, y) != background.pixel(0, y)}
        assert mask == native_branch_mask(style, button, slot, expanded, image)
        assert mask
        return mask

    second_mask = assert_branch(second, True)

    def assert_state(expanded):
        qt_app.processEvents()
        assert first.isChecked() == table(panel).isVisible() == expanded
        assert second.isChecked() and table(panel, 2).isVisible()
        assert assert_branch(second, True) == second_mask
        assert name.x() + name.contentsRect().left() == text_left
        return assert_branch(first, expanded)

    closed_mask = assert_state(False)
    assert text_left > slot.right()
    QTest.mouseClick(first, Qt.LeftButton, pos=slot.center())
    open_mask = assert_state(True)
    assert open_mask != closed_mask

    QTest.mouseClick(first, Qt.LeftButton, pos=QPoint(text_left + 3, slot.center().y()))
    assert assert_state(False) == closed_mask

    first.setFocus()
    assert first.hasFocus()
    QTest.keyClick(first, Qt.Key_Space)
    assert assert_state(True) == open_mask


@pytest.mark.parametrize(("entry", "error"), [
    ("not a pair", TypeError),
    (("缺少指标",), ValueError),
    ((None, ()), TypeError),
    (("工位", "指标"), TypeError),
    (("工位", ("指标", 2)), TypeError),
    ((" ", ()), ValueError),
])
def test_invalid_workbench_station_batch_preserves_display(panel, entry, error):
    panel.set_station_count(1)
    panel.set_station_row_count(1, 1)
    panel.set_station_row(1, 0, ("已显示", 10, 20))
    panel.set_station_results(1, "OK", "NG")
    snapshot = panel.get_station_data(1)

    with pytest.raises(error, match=r"stations\[1\]"):
        panel.show_workbench_stations((("待应用", ("新指标",)), entry))

    assert panel.get_station_count() == 1
    assert header(panel).accessibleName() == "工位 1"
    assert panel.get_station_data(1) == snapshot
    assert table(panel).item(0, 0).text() == "已显示"


def test_public_updates_and_independent_tables(panel):
    assert panel.set_station_count(2) is None
    assert panel.set_station_row_count(1, 2) is None
    panel.set_station_row_count(2, 1)
    assert panel.set_station_row(1, 0, ["SPL dB(A)", 68.5, 63.6]) is None
    assert panel.set_station_column(1, 2, [60, 61]) is None
    assert panel.set_station_cell(1, 1, 0, "响度 sone") is None
    assert panel.set_station_results(1, "ok", " NG ") is None
    snapshot = panel.get_station_data(1)
    assert snapshot == {"station_id": 1, "rows": [["SPL dB(A)", "68.5", "60"], ["响度 sone", "--", "61"]], "forward": "OK", "reverse": "NG", "result": "--"}
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
    assert panel.get_station_data(3) == {"station_id": 3, "rows": [], "forward": "--", "reverse": "--", "result": "--"}
    panel.set_station_row_count(1, 1)
    panel.set_station_row_count(1, 3)
    assert panel.get_station_data(1)["rows"] == [["kept", "--", "-3"], ["--"] * 3, ["--"] * 3]
    assert header(panel).isChecked()
    panel.set_station_row_count(2, 1)
    panel.set_station_results(2, "NG", "OK")
    panel.set_station_count(1)
    panel.set_station_count(2)
    assert panel.get_station_data(2) == {"station_id": 2, "rows": [], "forward": "--", "reverse": "--", "result": "--"}
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
    assert panel.get_station_data(1) == {"station_id": 1, "rows": [["a", "b", "2.0"], ["--", "--", "--"]], "forward": "--", "reverse": "--", "result": "--"}


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
@pytest.mark.parametrize("method,tail", [("set_station_row_count", (0,)), ("set_station_row", (0, [1, 2, 3])), ("set_station_column", (0, [1])), ("set_station_cell", (0, 0, 1)), ("set_station_results", ("OK", "NG")), ("get_station_data", ()), ("show_station_result", ("OK", "NG", "NG", [["a", 1, 2]]))])
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
    panel.set_station_results(1, "OK", "NG", result="NG")
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
    labels = header(panel).findChildren(QLabel)
    available = header(panel).contentsRect().width()
    assert labels[0].width() == int(available * .46)
    assert max(label.width() for label in labels[1:]) - min(label.width() for label in labels[1:]) <= 2
    for name in ("Forward", "Reverse", "Result"):
        general_label = panel.findChild(QLabel, "stationGeneral" + name)
        station_label = panel.findChild(QLabel, "station" + name + "1")
        assert abs(general_label.mapTo(panel, general_label.rect().center()).x() -
                   station_label.mapTo(panel, station_label.rect().center()).x()) <= 1
    general = panel.findChild(QLabel, "stationGeneralForward")
    forward = panel.findChild(QLabel, "stationForward1")
    reverse = panel.findChild(QLabel, "stationReverse1")
    result = panel.findChild(QLabel, "stationResult1")
    pending = panel.findChild(QLabel, "stationForward2")
    for label, color in [(forward, "#00824e"), (reverse, "#ca3245"), (pending, "#617183"), (result, "#ca3245")]:
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
            for name in ("Forward", "Reverse", "Result"):
                label = panel.findChild(QLabel, "stationGeneral" + name)
                station_label = panel.findChild(QLabel, "station" + name + "1")
                assert abs(label.mapTo(panel, label.rect().center()).x() -
                           station_label.mapTo(panel, station_label.rect().center()).x()) <= 1
    # Removing the scrollbar must restore the full column width too.
    panel.set_station_count(0)
    qt_app.processEvents()
    QTest.qWait(10)
    assert panel._general_header.contentsRect().width() == panel._scroll.viewport().width()


@pytest.mark.parametrize("status", ["--", "OK", "NG"])
def test_complete_result_snapshot_preserves_shape_fold_and_other_station(panel, status):
    panel.show_workbench_stations((("一号工位", ("SPL", "FFT")), ("二号工位", ("SPL",))))
    header(panel).setChecked(True)
    other = panel.get_station_data(2)
    rows = [["SPL", "62.30 dB(A)", "63.10 dB(A)"], ["FFT", "分析已完成", "--"]]
    assert panel.show_station_result(1, " ok ", "ng", status.lower(), rows) is None
    expected = {"station_id": 1, "forward": "OK", "reverse": "NG", "result": status,
                "rows": [row[:] for row in rows]}
    assert panel.get_station_data(1) == expected
    assert table(panel).columnCount() == 3
    assert table(panel).rowCount() == 2
    assert header(panel).isChecked()
    assert panel.get_station_data(2) == other
    assert panel.findChild(QLabel, "stationResult1").text() == status
    assert [[table(panel).item(r, c).text() for c in range(3)] for r in range(2)] == rows
    rows[0][0] = "mutated"
    rows.clear()
    snapshot = panel.get_station_data(1)
    snapshot["rows"][0][1] = "mutated"
    snapshot["result"] = "mutated"
    assert panel.get_station_data(1) == expected


@pytest.mark.parametrize("forward,reverse,result,rows,error", [
    ("pending", "NG", "NG", [["new", 1, 2], ["new", 3, 4]], ValueError),
    ("OK", None, "NG", [["new", 1, 2], ["new", 3, 4]], TypeError),
    ("OK", "NG", "pending", [["new", 1, 2], ["new", 3, 4]], ValueError),
    ("OK", "NG", None, [["new", 1, 2], ["new", 3, 4]], TypeError),
    ("OK", "NG", "NG", "not rows", TypeError),
    ("OK", "NG", "NG", [["new", 1, 2]], ValueError),
    ("OK", "NG", "NG", [["new", 1, 2], ["bad", 3]], ValueError),
    ("OK", "NG", "NG", [["new", 1, 2], ["bad", 3, True]], TypeError),
    ("OK", "NG", "NG", [["new", 1, 2], ["bad", 3, math.inf]], ValueError),
    ("OK", "NG", "NG", [["new", 1, 2], "bad"], TypeError),
])
def test_invalid_complete_result_is_atomic(panel, forward, reverse, result, rows, error):
    panel.show_workbench_stations((("工位", ("SPL", "FFT")), ("其他", ())))
    panel.set_station_results(1, "NG", "OK", result="OK")
    header(panel).setChecked(True)
    before = [panel.get_station_data(i) for i in (1, 2)]
    with pytest.raises(error, match=".+"):
        panel.show_station_result(1, forward, reverse, result, rows)
    assert [panel.get_station_data(i) for i in (1, 2)] == before
    assert header(panel).isChecked()
    assert table(panel).rowCount() == 2 and table(panel).columnCount() == 3
    assert [[table(panel).item(r, c).text() for c in range(3)] for r in range(2)] == before[0]["rows"]
    for name, value in (("Forward", "NG"), ("Reverse", "OK"), ("Result", "OK")):
        assert panel.findChild(QLabel, f"station{name}1").text() == value


def test_zero_rows_and_rebuilt_stations_reset_final_result(panel):
    panel.show_workbench_stations((("工位", ()),))
    assert panel.get_station_data(1)["result"] == "--"
    panel.show_station_result(1, "OK", "--", "OK", ())
    assert panel.get_station_data(1)["result"] == "OK"
    assert table(panel).rowCount() == 0 and table(panel).columnCount() == 3
    panel.set_station_results(1, "NG", "OK")  # Original call form remains legal.
    assert panel.get_station_data(1)["result"] == "--"
    panel.show_station_result(1, "OK", "--", "OK", [])
    panel.show_workbench_stations((("工位", ()), ("工位二", ())))
    assert [panel.get_station_data(i)["result"] for i in (1, 2)] == ["--", "--"]
    panel.show_station_result(2, "NG", "OK", "NG", [])
    panel.show_workbench_stations(())
    panel.show_workbench_stations((("新工位", ()),))
    assert panel.get_station_data(1)["result"] == "--"


@pytest.mark.parametrize("bad,error,field", [
    (["a", 1], ValueError, r"station_id 1 rows\[1\]"),
    ("row", TypeError, r"station_id 1 rows\[1\]"),
    (["a", 1, object()], TypeError, r"station_id 1 rows\[1\]\[2\]"),
    (["a", 1, math.nan], ValueError, r"station_id 1 rows\[1\]\[2\]"),
])
def test_complete_result_contract_errors_identify_station_and_row(panel, bad, error, field):
    panel.show_workbench_stations((("station", ("SPL", "FFT")),))
    before = panel.get_station_data(1)
    with pytest.raises(error, match=field):
        panel.show_station_result(1, "OK", "NG", "NG", [["SPL", 1, 2], bad])
    assert panel.get_station_data(1) == before
