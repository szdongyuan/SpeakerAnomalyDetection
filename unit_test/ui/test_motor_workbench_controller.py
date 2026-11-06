from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QAbstractButton, QComboBox, QLabel

from base.motor_workbench_config import (
    WorkbenchCondition,
    WorkbenchConfig,
    WorkbenchConfigurationError,
    WorkbenchPlotCell,
    WorkbenchPlotLayout,
    WorkbenchQueueConfig,
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
    project_id = "product"
    result_directory = "results"

    def condition(condition_id, name, queue_name, station_ids, metrics):
        acquisition = MappingProxyType({
            "detail": MappingProxyType({"sample_rate": 44100, "total_time": 2}),
        })
        analysis = MappingProxyType({
            "display_sequence": tuple(metrics), **{
                name: MappingProxyType(item) for name, item in metrics.items()
            },
        })
        seq1 = MappingProxyType({"acq": acquisition, "analysis_list": analysis})
        queue = WorkbenchQueueConfig(
            sequence_config=(MappingProxyType({"seq1": seq1}),),
            acquisition=acquisition,
            acquisition_mode="RECORD_ONLY",
            analysis=analysis,
            metric_names=tuple(metrics),
        )
        saved = MappingProxyType({
            "condition_id": condition_id,
            "condition_name": name,
            "test_queue": queue_name,
            "applicable_station_ids": station_ids,
            "trigger_state": "",
            "segmented_analysis": MappingProxyType({"mode": "none"}),
        })
        return WorkbenchCondition(
            **saved, queue=queue, project_id=project_id, result_root_directory=result_directory,
            condition_config=saved,
        )

    return WorkbenchConfig(
        project_id=project_id,
        project_name="电机",
        result_root_directory=result_directory,
        stations=(
            WorkbenchStation("right", "右工位", ()),
            WorkbenchStation("left", "左工位", ()),
            WorkbenchStation("spare", "备用工位", ()),
        ),
        conditions=(
            condition("warm", "热机", "queue-a", ("left", "right"), {"响度": {}, "": {}, "峰值": {}}),
            condition("cold", "冷机", "queue-b", ("right",), {"峰值": {}, "转速": {}}),
            condition("empty", "空指标", "queue-c", ("left",), {}),
        ),
        active_file="product.json",
        plot_layout=WorkbenchPlotLayout(1, 3, ((WorkbenchPlotCell("声压级", "响度", "SPL"),) * 3,)),
    )


def _plot_condition(source, condition_id, name, queue_name, station_ids, items, *, extra=None):
    analysis = MappingProxyType({
        "display_sequence": tuple(items),
        **{key: MappingProxyType(value) for key, value in (items | (extra or {})).items()},
    })
    seq1 = MappingProxyType({"acq": source.queue.acquisition, "analysis_list": analysis})
    queue = replace(
        source.queue, analysis=analysis, metric_names=tuple(items),
        sequence_config=(MappingProxyType({"seq1": seq1}),),
    )
    changes = {
        "condition_id": condition_id,
        "condition_name": name,
        "test_queue": queue_name,
        "applicable_station_ids": station_ids,
    }
    return replace(
        source, **changes, queue=queue,
        condition_config=MappingProxyType(dict(source.condition_config) | changes),
    )


@pytest.fixture
def plot_config(config):
    source = config.conditions[0]
    first = _plot_condition(source, "c1", "工况1", "Queue A", ("s1",), {
        "声压级": {"type": " SPL ", "weighting": "a"},
        "第一有": {"type": "FFT", "weighting": "None"},
        "差值": {"type": "FFT", "weighting": "C", "baseline_display_mode": "delta"},
    })
    second = _plot_condition(source, "c2", "工况2", "Queue B", ("s2", "s1"), {
        "声压级": {"type": "SPL"},
        "第二有": {"type": "FFT", "weighting": "c"},
        "差值": {"type": "FFT", "baseline_display_mode": "delta"},
        "类型不同": {"type": "SPL"},
    }, extra={"第一有": {"type": "FFT", "weighting": "C"}})
    third = _plot_condition(source, "c3", "工况3", "Queue C", ("s1",), {
        "声压级": {"type": "SPL", "weighting": "A"},
    })
    fourth = _plot_condition(source, "c4", "工况4", "Queue D", (), {})
    rows = (
        (WorkbenchPlotCell("声压级图", "声压级", "SPL"),
         WorkbenchPlotCell("单工况图", "声压级", "SPL"),
         WorkbenchPlotCell("无工况图", "声压级", "SPL"),
         WorkbenchPlotCell("空列图", "声压级", "SPL")),
        (WorkbenchPlotCell("第一有图", "第一有", "FFT"),
         WorkbenchPlotCell("类型不同图", "类型不同", "FFT"),
         WorkbenchPlotCell("无工况图", "声压级", "SPL"),
         WorkbenchPlotCell("空列图", "声压级", "SPL")),
        (WorkbenchPlotCell("第二有图", "第二有", "FFT"),
         WorkbenchPlotCell("差值图", "差值", "FFT"),
         WorkbenchPlotCell("无工况图", "声压级", "SPL"),
         WorkbenchPlotCell("空列图", "声压级", "SPL")),
    )
    return replace(
        config,
        stations=tuple(WorkbenchStation(f"s{i}", f"工位{i}", ()) for i in (1, 2, 3)),
        conditions=(first, second, third, fourth),
        plot_layout=WorkbenchPlotLayout(3, 4, rows),
    )


@pytest.fixture
def workbench(qt_app, monkeypatch):
    calls = []
    products = []
    video_requests = []
    repository = RepositoryStub(calls)
    panel = MotorStationTestPanel()
    grid = StationPlotGrid()
    plot_batches = []

    def record_call(view, method, name):
        render = getattr(view, method)

        def record_and_render(*data):
            calls.append(name)
            render(*data)
            if name == "grid":
                plot_batches.append(data)

        monkeypatch.setattr(view, method, record_and_render)

    record_call(panel, "show_workbench_stations", "panel")
    record_call(grid, "show_workbench_layout", "grid")
    controller = MotorWorkbenchController(
        repository, panel, grid, products.append,
        lambda: video_requests.append("settings"),
    )
    yield SimpleNamespace(
        controller=controller, repository=repository, panel=panel, grid=grid,
        calls=calls, products=products, video_requests=video_requests, plot_batches=plot_batches,
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
    assert set(plot_cells(workbench.grid)) == {(0, c) for c in range(3)}
    assert all(not column_options(workbench.grid, i).isEnabled() for i in range(3))
    expected_names = (("响度", "", "峰值（热机）", "峰值（冷机）", "转速"),
                      ("响度", "", "峰值"), ())
    for station, metrics in enumerate(expected_names, start=1):
        data = workbench.panel.get_station_data(station)
        assert (data["forward"], data["reverse"]) == ("--", "--")
        assert data["rows"] == [[metric, "--", "--"] for metric in metrics]


def test_refresh_failure_and_equal_recovery_keep_existing_results(workbench, config, caplog):
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
    assert "product.json: unknown station" in caplog.text


def test_plot_projection_preserves_positions_and_all_condition_queries(workbench, plot_config):
    config = plot_config
    controller = workbench.controller
    workbench.repository.result = config
    controller.refresh_configuration()
    names, cells = workbench.plot_batches[-1]
    assert names == ("工位1", "工位2", "工位3", "")

    spl = cells[0][0]
    assert spl.unit == "dB(A)"
    assert (spl.first_condition, spl.second_condition) == ("工况1", "")
    assert spl.warning == "单位不一致"
    assert "工况1" in spl.warning_detail and "dB(A)" in spl.warning_detail
    assert "工况2" in spl.warning_detail and "dB(Z)" in spl.warning_detail
    first, second = controller.get_plot_bindings(0, 0)
    assert (first.station_id, first.condition_id, first.analysis_item, first.analysis_type) == (
        "s1", "c1", "声压级", "SPL",
    )
    assert second is None

    for row, column, expected in (
        (0, 1, ("dB(Z)", "工况2", "", ("c2", None))),
        (0, 2, ("", "", "", (None, None))),
        (0, 3, ("", "", "", (None, None))),
        (1, 0, ("dB SPL", "工况1", "工况2", ("c1", None))),
        (1, 1, ("", "工况2", "", (None, None))),
        (2, 0, ("dB(C) SPL", "工况1", "工况2", (None, "c2"))),
        (2, 1, ("dB", "工况2", "", ("c2", None))),
    ):
        cell = cells[row][column]
        bindings = controller.get_plot_bindings(row, column)
        condition_ids = tuple(binding.condition_id if binding is not None else None for binding in bindings)
        assert (cell.unit, cell.first_condition, cell.second_condition, condition_ids) == expected

    assert cells[0][2].title == "无工况图"
    assert cells[0][3].title == cells[0][3].warning == ""
    assert "工况2" in cells[1][0].warning_detail
    assert "类型不符" in cells[1][1].warning_detail
    assert "工况1" in cells[2][0].warning_detail
    for row, column in ((-1, 0), (0, -1), (3, 0), (0, 4)):
        assert controller.get_plot_bindings(row, column) == (None, None)

    condition = controller.get_condition_config("c2")
    assert condition.test_queue == "Queue B"
    assert condition.project_id == config.project_id
    assert condition.result_root_directory == config.result_root_directory
    for item in config.conditions:
        assert controller.get_condition_config(item.condition_id) is item
    assert controller.get_station_channels("s1") is config.stations[0].channels
    assert controller.get_station_channels("missing") is None
    assert controller.get_condition_config("missing") is None


@pytest.mark.parametrize(("weighting", "mode", "invalid_slot"), [
    (" A ", "overlay", 0),
    ("B", "delta", 0),
    ("B", "delta", 1),
])
def test_invalid_fft_weighting_preserves_valid_sibling(workbench, config, weighting, mode, invalid_slot):
    item = "频谱"
    station = config.stations[0]
    invalid = {"type": "FFT", "weighting": weighting, "baseline_display_mode": mode}
    valid = {"type": "FFT", "weighting": "a"}
    items = (invalid, valid) if invalid_slot == 0 else (valid, invalid)
    conditions = tuple(
        _plot_condition(config.conditions[0], f"c{i}", f"工况{i}", f"queue-{i}",
                        (station.station_id,), {item: analysis})
        for i, analysis in enumerate(items, start=1)
    )
    layout = WorkbenchPlotLayout(1, 1, ((WorkbenchPlotCell(item, item, "FFT"),),))
    workbench.repository.result = replace(
        config, stations=(station,), conditions=conditions, plot_layout=layout,
    )
    controller = workbench.controller
    controller.refresh_configuration()

    display = workbench.plot_batches[-1][1][0][0]
    assert (display.unit, display.first_condition, display.second_condition) == (
        "dB(A) SPL", "工况1", "工况2",
    )
    assert display.warning == "分析项不可用"
    assert conditions[invalid_slot].condition_name in display.warning_detail
    assert "计权" in display.warning_detail and repr(weighting) in display.warning_detail
    bindings = controller.get_plot_bindings(0, 0)
    assert bindings[invalid_slot] is None
    assert bindings[1 - invalid_slot].condition_id == conditions[1 - invalid_slot].condition_id
    for condition, analysis in zip(conditions, items):
        assert controller.get_condition_config(condition.condition_id) is condition
        assert condition.export_process_config()["analysis_config"][item] == analysis


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
