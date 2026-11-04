"""Real Qt station rendering and fold interaction after task-browser removal."""
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QLabel, QPushButton, QTableWidget

from ui.sequence.motor_left_panel import MotorDetectionLeftPanel


@pytest.fixture
def panel(qt_app):
    widget = MotorDetectionLeftPanel(None, condition_configs=[
        {"key": "a", "condition_name": "a"}, {"key": "b", "condition_name": "b"},
    ], queue_catalog={})
    widget.setAttribute(Qt.WA_DontShowOnScreen)
    widget.set_station_count(2)
    widget.set_station_row_count(1, 2)
    widget.set_station_row(1, 0, ["metric", 12, 34])
    widget.resize(600, 800)
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


@pytest.mark.parametrize("forward,reverse", [("--", "--"), ("OK", "NG"), ("NG", "OK")])
def test_station_status_palette_and_real_text_pixels(panel, qt_app, tmp_path, forward, reverse):
    panel.set_station_results(1, forward, reverse)
    qt_app.processEvents()
    colors = {"--": "#617183", "OK": "#00824e", "NG": "#ca3245"}
    for name, value in [("stationForward1", forward), ("stationReverse1", reverse)]:
        label = panel.station_panel.findChild(QLabel, name)
        assert label.text() == value
        assert label.palette().color(label.foregroundRole()) == QColor(colors[value])
        image = label.grab().toImage()
        expected = QColor(colors[value]).getRgb()[:3]
        closest = min(max(abs(a-b) for a,b in zip(image.pixelColor(x,y).getRgb()[:3], expected))
                      for y in range(image.height()) for x in range(image.width()))
        assert closest <= 35
        assert image.save(str(tmp_path / f"{name}-{value}.png"))
    assert panel.station_panel.findChild(QLabel, "stationForward2").text() == "--"


def test_station_fold_is_independent_of_backend_recording_selection(panel, qt_app):
    selected = []
    panel.condition_selected.connect(selected.append)
    first = panel.station_panel.findChild(QPushButton, "stationHeader1")
    second = panel.station_panel.findChild(QPushButton, "stationHeader2")
    first.click()
    second.click()
    qt_app.processEvents()
    assert first.isChecked() and second.isChecked()
    assert panel.station_panel.findChild(QTableWidget, "stationTable1").isVisible()
    panel.set_condition_result("b", "采集中", "running")
    assert selected == ["b"]
    assert panel.condition_state.rows["b"]["result"] == "采集中"
    assert first.isChecked() and second.isChecked()
    first.click()
    assert not first.isChecked() and second.isChecked()
    assert selected == ["b"]
    assert panel.get_station_data(1)["rows"][0] == ["metric", "12", "34"]


def test_station_table_actual_alternating_rows_and_fixed_headers(panel, qt_app):
    panel.station_panel.findChild(QPushButton, "stationHeader1").click()
    qt_app.processEvents()
    table = panel.station_panel.findChild(QTableWidget, "stationTable1")
    assert [table.horizontalHeaderItem(c).text() for c in range(3)] == ["指标", "正转", "反转"]
    assert not table.verticalHeader().isVisible()
    image = table.viewport().grab().toImage()
    x = table.viewport().width() // 2
    assert image.pixelColor(x, table.rowViewportPosition(0)+3) == QColor("#ffffff")
    assert image.pixelColor(x, table.rowViewportPosition(1)+3) == QColor("#fafbfd")
