from dataclasses import replace
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QAbstractButton, QComboBox, QLabel

from base.motor_workbench_config import (
    WorkbenchCondition,
    WorkbenchConfig,
    WorkbenchConfigurationError,
    WorkbenchStation,
)
from base.video.models import VideoStatus
from ui.sequence.motor_station_test_panel import MotorStationTestPanel
from ui.sequence.motor_workbench_controller import MotorWorkbenchController
from ui.sequence.station_plot_cell import StationPlotCell
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.video_monitor_widget import VideoMonitorWidget


class RepositoryStub:
    def __init__(self, calls):
        self.result = None
        self.calls = calls

    def load_active(self):
        self.calls.append("load")
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def config():
    return WorkbenchConfig(
        project_id="product",
        project_name="电机",
        result_root_directory="results",
        stations=(
            WorkbenchStation("right", "右工位", ()),
            WorkbenchStation("left", "左工位", ()),
            WorkbenchStation("spare", "备用工位", ()),
        ),
        conditions=(
            WorkbenchCondition("warm", "热机", "queue-a", ("left", "right"), "",
                               {"mode": "none"}, ("响度", "", "峰值")),
            WorkbenchCondition("cold", "冷机", "queue-b", ("right",), "",
                               {"mode": "none"}, ("峰值", "转速")),
            WorkbenchCondition("empty", "空指标", "queue-c", ("left",), "",
                               {"mode": "none"}, ()),
        ),
        active_file="product.json",
    )


@pytest.fixture
def workbench(qt_app, monkeypatch):
    calls = []
    products = []
    video_requests = []
    repository = RepositoryStub(calls)
    panel = MotorStationTestPanel()
    grid = StationPlotGrid()

    def record_call(view, method, name):
        render = getattr(view, method)

        def record_and_render(data):
            calls.append(name)
            render(data)

        monkeypatch.setattr(view, method, record_and_render)

    record_call(panel, "show_workbench_stations", "panel")
    record_call(grid, "show_workbench_columns", "grid")
    controller = MotorWorkbenchController(
        repository, panel, grid, products.append,
        lambda: video_requests.append("settings"),
    )
    yield SimpleNamespace(
        controller=controller, repository=repository, panel=panel, grid=grid,
        calls=calls, products=products, video_requests=video_requests,
    )
    for view in (panel, grid):
        view.close()
        view.deleteLater()
    qt_app.processEvents()


def station_header(panel, number):
    return panel.findChild(QAbstractButton, f"stationHeader{number}")


def column_options(grid, column):
    return grid.findChild(QComboBox, f"stationOptions{column}")


def plot_cells(grid):
    return {(cell.row, cell.column): cell for cell in grid.findChildren(StationPlotCell)}


@pytest.mark.parametrize("initial_error", [None, "queue.json: missing display_sequence"])
def test_first_refresh_starts_empty_and_applies_valid_configuration(workbench, config, initial_error):
    if initial_error is not None:
        workbench.repository.result = WorkbenchConfigurationError(initial_error)
    workbench.controller.refresh_configuration()

    assert workbench.calls == ["panel", "grid", "load"]
    assert workbench.panel.get_station_count() == 0
    assert not plot_cells(workbench.grid)
    assert workbench.products == [""]

    workbench.repository.result = config
    workbench.controller.refresh_configuration()

    names = ["右工位", "左工位", "备用工位"]
    assert workbench.calls == ["panel", "grid", "load", "load", "panel", "grid"]
    assert workbench.products == ["", "电机"]
    assert workbench.panel.get_station_count() == 3
    assert [station_header(workbench.panel, i).accessibleName() for i in (1, 2, 3)] == names
    assert [workbench.grid.findChild(QLabel, f"stationName{i}").toolTip() for i in range(3)] == names
    assert set(plot_cells(workbench.grid)) == {(r, c) for r in range(2) for c in range(3)}
    assert all(not column_options(workbench.grid, i).isEnabled() for i in range(3))
    expected_names = (("响度", "", "峰值（热机）", "峰值（冷机）", "转速"),
                      ("响度", "", "峰值"), ())
    for station, metrics in enumerate(expected_names, start=1):
        data = workbench.panel.get_station_data(station)
        assert (data["forward"], data["reverse"]) == ("--", "--")
        assert data["rows"] == [[metric, "--", "--"] for metric in metrics]


def test_refresh_failure_and_equal_recovery_keep_existing_results(workbench, config):
    workbench.repository.result = config
    workbench.controller.refresh_configuration()
    workbench.panel.set_station_results(1, "OK", "NG")
    workbench.panel.set_station_cell(1, 0, 1, 12)
    workbench.grid.set_curve_data(0, 0, [{"x": [1, 2], "y": [3, 4]}])
    workbench.grid.set_column_options(0, [("结果", 42)])
    workbench.calls.clear()

    expected_data = workbench.panel.get_station_data(1)
    plot = plot_cells(workbench.grid)[0, 0].plot_widget.getPlotItem()
    curves = plot.listDataItems()
    assert len(curves) == 1

    for result in (WorkbenchConfigurationError("product.json: unknown station"),
                   replace(config), replace(config)):
        workbench.repository.result = result
        workbench.controller.refresh_configuration()
        assert workbench.panel.get_station_data(1) == expected_data
        assert plot.listDataItems() == curves
        assert column_options(workbench.grid, 0).currentData() == 42

    assert workbench.calls == ["load", "load", "load"]
    assert workbench.products == ["", "电机"]


def test_switch_resets_same_size_views_and_clearing_is_idempotent(workbench, config):
    workbench.repository.result = config
    workbench.controller.refresh_configuration()
    workbench.panel.set_station_results(1, "OK", "NG")
    workbench.panel.set_station_cell(1, 0, 1, 12)
    workbench.grid.set_curve_data(0, 0, [{"x": [1, 2], "y": [3, 4]}])
    workbench.grid.set_column_options(0, [("结果", 42)])
    workbench.calls.clear()

    workbench.repository.result = replace(
        config, project_name="风扇", active_file="fan.json",
        stations=(replace(config.stations[0], station_name="前工位"), *config.stations[1:]),
    )
    workbench.controller.refresh_configuration()

    assert workbench.calls == ["load", "panel", "grid"]
    assert workbench.products == ["", "电机", "风扇"]
    assert station_header(workbench.panel, 1).accessibleName() == "前工位"
    assert workbench.grid.findChild(QLabel, "stationName0").toolTip() == "前工位"
    data = workbench.panel.get_station_data(1)
    assert (data["forward"], data["reverse"]) == ("--", "--")
    assert data["rows"][0] == ["响度", "--", "--"]
    assert not plot_cells(workbench.grid)[0, 0].plot_widget.getPlotItem().listDataItems()
    assert column_options(workbench.grid, 0).count() == 0

    workbench.repository.result = None
    workbench.controller.refresh_configuration()
    workbench.controller.refresh_configuration()
    assert workbench.calls == ["load", "panel", "grid", "load", "panel", "grid", "load"]
    assert workbench.products == ["", "电机", "风扇", ""]
    assert workbench.panel.get_station_count() == 0
    assert not plot_cells(workbench.grid)


def test_unexpected_repository_error_propagates(workbench):
    workbench.repository.result = RuntimeError("unexpected load error")
    with pytest.raises(RuntimeError, match="unexpected load error"):
        workbench.controller.refresh_configuration()


def test_local_view_actions_stay_local_and_program_updates_emit_no_intent(workbench, config):
    workbench.repository.result = config
    workbench.controller.refresh_configuration()
    selection = QSignalSpy(workbench.grid.column_selection_changed)
    header = station_header(workbench.panel, 1)
    header_click = QSignalSpy(header.clicked)
    before = workbench.panel.get_station_data(1)
    workbench.calls.clear()

    workbench.grid.set_column_options(0, [("第一项", 1), ("第二项", 2)])
    workbench.grid.set_column_current_index(0, 1)
    assert len(selection) == 0
    QTest.keyClick(column_options(workbench.grid, 0), Qt.Key_Up)
    assert list(selection) == [[0, 0, 1]]
    assert len(header_click) == 0
    assert workbench.panel.get_station_data(1) == before

    QTest.mouseClick(header, Qt.LeftButton)
    assert len(header_click) == 1
    assert len(selection) == 1
    assert column_options(workbench.grid, 0).currentData() == 1
    assert workbench.calls == []

    workbench.repository.result = replace(config, project_name="另一产品")
    workbench.controller.refresh_configuration()
    assert workbench.calls == ["load", "panel", "grid"]
    assert len(selection) == 1
    assert len(header_click) == 1
    assert not column_options(workbench.grid, 0).isEnabled()


def test_settings_intent_calls_only_the_video_port_once(workbench):
    class PreviewBridge(QObject):
        status_changed = pyqtSignal(object)
        preview_changed = pyqtSignal(object)
        preview_requested = False
        service = SimpleNamespace(status=VideoStatus(connection="ready"))

    widget = VideoMonitorWidget(PreviewBridge())
    try:
        intents = QSignalSpy(widget.settings_requested)
        widget.settings_requested.connect(workbench.controller.open_video_settings)
        QTest.mouseClick(widget.settings_button, Qt.LeftButton)
        assert len(intents) == 1
        assert workbench.video_requests == ["settings"]
        assert workbench.calls == []
    finally:
        widget.close()
        widget.deleteLater()
