import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QAbstractButton, QComboBox, QLabel

from base.workbench_config_values import WorkbenchChannel, WorkbenchHardwareChannel, WorkbenchCondition, WorkbenchContext, WorkbenchStation
from base.workbench_recording_archive import WorkbenchRecordingArchiveError
from base.workbench_configuration_loader import WorkbenchConfiguration, WorkbenchConfigurationError
from base.workbench_recording_config import WorkbenchRecordingConfig
from base.workbench_analysis_config import WorkbenchAnalysisConfig, AnalysisTestQueue
from base.workbench_ui_config import WorkbenchUiConfig, WorkbenchPlotCell, WorkbenchPlotLayout
from base.recording_process_protocol import FrozenConfig, RecordingCancelled, RecordingFailure
from base.video.models import VideoStatus
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from ui.sequence.motor_station_test_panel import MotorStationTestPanel
from ui.sequence.motor_workbench_controller import MotorWorkbenchController
from ui.sequence.station_plot_cell import StationPlotCell
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.video_monitor_widget import VideoMonitorWidget
from ui.workbench_recording_device import WorkbenchRecordingDeviceSnapshot


class RepositoryStub:
    def __init__(self, calls):
        self.source_identity = "product.json"
        self.result = None
        self.calls = calls
        self.catalog = (("product.json", "电机"), ("second.json", "另一电机"))
        self.candidate = None

    def load_product_catalog(self):
        self.calls.append("catalog")
        if isinstance(self.catalog, Exception):
            raise self.catalog
        return self.catalog

    def load_candidate(self, file_name):
        self.calls.append(("candidate", file_name))
        if isinstance(self.candidate, Exception):
            raise self.candidate
        return self.candidate

    def accept_product_source(self, file_name):
        self.calls.append(("accept", file_name))
        self.source_identity = str(Path("products") / file_name)

    def load_active(self):
        self.calls.append("load")
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def configuration_with(config, *, plot_layout=None, test_queues=None, parameters=None, **context_changes):
    context = replace(config.recording.context, **context_changes)
    return WorkbenchConfiguration(
        WorkbenchRecordingConfig(context, parameters if parameters is not None else config.recording.parameters_by_queue),
        WorkbenchAnalysisConfig(context, test_queues if test_queues is not None else config.analysis.test_queues),
        WorkbenchUiConfig(plot_layout or config.ui.layout, {c.test_queue: "cumulative" for c in context.conditions}),
    )


@pytest.fixture
def config(tmp_path):
    conditions = (
        WorkbenchCondition("warm", "热机", "queue-a", ("left", "right"), "", {"mode": "none"}),
        WorkbenchCondition("cold", "冷机", "queue-b", ("right",), "", {"mode": "none"}),
        WorkbenchCondition("empty", "空指标", "queue-c", ("left",), "", {"mode": "none"}),
    )
    context = WorkbenchContext("product", "电机", str(tmp_path / "results"), (
        WorkbenchStation("right", "右工位", ()), WorkbenchStation("left", "左工位", ()),
        WorkbenchStation("spare", "备用工位", ()),
    ), conditions)
    test_queues = {name: AnalysisTestQueue(None, {"display_sequence": tuple(items), **items}) for name, items in (
        ("queue-a", {"响度": {}, "": {}, "峰值": {}}),
        ("queue-b", {"峰值": {}, "转速": {}}), ("queue-c", {}),
    )}
    return WorkbenchConfiguration(
        WorkbenchRecordingConfig(context, {c.test_queue: {"total_time": 2} for c in conditions}),
        WorkbenchAnalysisConfig(context, test_queues),
        WorkbenchUiConfig(WorkbenchPlotLayout(1, 3, ((WorkbenchPlotCell("声压级", "响度", "SPL"),) * 3,)), {c.test_queue: "cumulative" for c in conditions}),
    )


def _plot_condition(test_queues, source, condition_id, name, queue_name, station_ids, items, *, extra=None):
    test_queues[queue_name] = AnalysisTestQueue(None, {"display_sequence": tuple(items), **items, **(extra or {})})
    return replace(source, condition_id=condition_id, condition_name=name,
                   test_queue=queue_name, applicable_station_ids=station_ids)


@pytest.fixture
def plot_config(config):
    test_queues = {}
    source = config.recording.context.conditions[0]
    first = _plot_condition(test_queues, source, "c1", "工况1", "Queue A", ("s1",), {
        "声压级": {"type": " SPL ", "weighting": "a"},
        "第一有": {"type": "FFT", "weighting": "None"},
        "差值": {"type": "FFT", "weighting": "C", "baseline_display_mode": "delta"},
    })
    second = _plot_condition(test_queues, source, "c2", "工况2", "Queue B", ("s2", "s1"), {
        "声压级": {"type": "SPL"},
        "第二有": {"type": "FFT", "weighting": "c"},
        "差值": {"type": "FFT", "baseline_display_mode": "delta"},
        "类型不同": {"type": "SPL"},
    }, extra={"第一有": {"type": "FFT", "weighting": "C"}})
    third = _plot_condition(test_queues, source, "c3", "工况3", "Queue C", ("s1",), {
        "声压级": {"type": "SPL", "weighting": "A"},
    })
    fourth = _plot_condition(test_queues, source, "c4", "工况4", "Queue D", (), {})
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
    return configuration_with(
        config,
        stations=tuple(WorkbenchStation(f"s{i}", f"工位{i}", ()) for i in (1, 2, 3)),
        conditions=(first, second, third, fourth), test_queues=test_queues,
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
                   configuration_with(config), configuration_with(config)):
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
        (1, 0, ("dB(Z)", "工况1", "工况2", ("c1", None))),
        (1, 1, ("", "工况2", "", (None, None))),
        (2, 0, ("dB(C)", "工况1", "工况2", (None, "c2"))),
        (2, 1, ("dB(Z)", "工况2", "", ("c2", None))),
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
    assert controller.get_recording_config() is config.recording
    assert controller.get_analysis_config() is config.analysis
    assert controller.get_ui_config() is config.ui
    for item in config.recording.context.conditions:
        assert controller.get_condition_config(item.condition_id) is item
    assert controller.get_station_channels("s1") is config.recording.context.stations[0].channels
    assert controller.get_station_channels("missing") is None
    assert controller.get_condition_config("missing") is None


def test_fft_delta_plot_warns_when_weighting_differs(workbench, plot_config):
    cell = WorkbenchPlotCell("差值图", "差值", "FFT")
    layout = WorkbenchPlotLayout(1, 4, ((cell,) * 4,))
    workbench.repository.result = configuration_with(plot_config, plot_layout=layout)
    workbench.controller.refresh_configuration()

    display = workbench.plot_batches[-1][1][0][0]
    assert display.warning == "单位不一致"
    assert display.warning_detail.splitlines() == ["工况1 dB(C)", "工况2 dB(Z)"]
    assert (display.unit, display.first_condition, display.second_condition) == ("dB(C)", "工况1", "")
    first, second = workbench.controller.get_plot_bindings(0, 0)
    assert first.condition_id == "c1"
    assert second is None


@pytest.mark.parametrize(("analysis_type", "first", "second", "unit"), [
    pytest.param("SPL", {"weighting": "a"}, {"weighting": "A"}, "dB(A)", id="spl-a"),
    pytest.param("SPL", {"weighting": "c"}, {"weighting": "C"}, "dB(C)", id="spl-c"),
    pytest.param("SPL", {"weighting": "None"}, {"weighting": "Z"}, "dB(Z)", id="spl-none"),
    pytest.param("SPL", {"weighting": "Z（NONE）"}, {"weighting": "Z"}, "dB(Z)", id="spl-z-none"),
    pytest.param("SPL", {}, {"weighting": "Z"}, "dB(Z)", id="spl-default"),
    pytest.param("FFT", {"weighting": "a"}, {"weighting": "A"}, "dB(A)", id="fft-a"),
    pytest.param("FFT", {"weighting": "c"}, {"weighting": "C"}, "dB(C)", id="fft-c"),
    pytest.param("FFT", {"weighting": "None"}, {"weighting": "Z"}, "dB(Z)", id="fft-none"),
    pytest.param("FFT", {"weighting": "Z（NONE）"}, {"weighting": "Z"}, "dB(Z)", id="fft-z-none"),
    pytest.param("FFT", {}, {"weighting": "Z"}, "dB(Z)", id="fft-default"),
    pytest.param("FFT", {"weighting": "C", "baseline_display_mode": "overlay"},
                 {"weighting": "C", "baseline_display_mode": "delta"}, "dB(C)", id="fft-mixed-modes"),
    pytest.param("FFT", {"weighting": "B", "baseline_display_mode": "delta"},
                 {"weighting": "B", "baseline_display_mode": "delta"}, "dB(B)", id="fft-b"),
])
def test_shared_plot_formats_weighting_and_preserves_raw_config(workbench, config, analysis_type, first, second, unit):
    item = "分析项"
    station = config.recording.context.stations[0]
    items = ({"type": analysis_type, **first}, {"type": analysis_type, **second})
    test_queues = {}
    conditions = tuple(
        _plot_condition(test_queues, config.recording.context.conditions[0], f"c{i}", f"工况{i}", f"queue-{i}",
                        (station.station_id,), {item: analysis})
        for i, analysis in enumerate(items, start=1)
    )
    layout = WorkbenchPlotLayout(1, 1, ((WorkbenchPlotCell(item, item, analysis_type),),))
    workbench.repository.result = configuration_with(
        config, stations=(station,), conditions=conditions, plot_layout=layout, test_queues=test_queues,
    )
    controller = workbench.controller
    controller.refresh_configuration()

    display = workbench.plot_batches[-1][1][0][0]
    assert (display.unit, display.first_condition, display.second_condition) == (
        unit, "工况1", "工况2",
    )
    assert display.warning == display.warning_detail == ""
    bindings = controller.get_plot_bindings(0, 0)
    assert tuple(binding.condition_id for binding in bindings) == ("c1", "c2")
    for condition, analysis in zip(conditions, items):
        assert controller.get_condition_config(condition.condition_id) is condition
        exported = controller.get_analysis_config().describe()
        assert exported["test_queues"][condition.test_queue]["analysis_config"][item] == analysis


def test_switch_resets_same_size_views_and_clearing_is_idempotent(workbench, config):
    workbench.repository.result = config
    workbench.controller.refresh_configuration()
    workbench.panel.set_station_results(1, "OK", "NG")
    workbench.panel.set_station_cell(1, 0, 1, 12)
    workbench.grid.set_curve_data(0, 0, [{"x": [1, 2], "y": [3, 4]}])
    workbench.grid.set_column_options(0, [("结果", 42)])
    workbench.calls.clear()

    workbench.repository.result = configuration_with(
        config, project_name="风扇",
        stations=(replace(config.recording.context.stations[0], station_name="前工位"), *config.recording.context.stations[1:]),
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

    workbench.repository.result = configuration_with(config, project_name="另一产品")
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


@pytest.fixture
def recording_workbench(workbench, config):
    from ui.sequence.motor_workbench_controller import WorkbenchRecordingPorts

    channel = WorkbenchChannel("input", "声音", "audio", WorkbenchHardwareChannel("default", 0))
    stations = tuple(replace(station, channels=(channel,)) for station in config.recording.context.stations)
    config = configuration_with(config, stations=stations, conditions=config.recording.context.conditions[:2])
    workbench.repository.result = config
    device = SimpleNamespace(
        login_ready=True, detail="准备中",
        snapshot=WorkbenchRecordingDeviceSnapshot("probe", 1, FrozenConfig.snapshot({
            "backend": "sounddevice", "name": "Test", "index": 0, "hostapi": 0,
            "max_input_channels": 2,
        }), (0, 1), 48000),
    )
    requests, callbacks, states, problems, completions = [], [], [], [], []
    cleanup_problems, archive_problems, archive_submissions = [], [], []
    service = SimpleNamespace(can_start_recording=True)
    outcome = SimpleNamespace(error=None, on_start=None, hardware_available=True, on_archive=None)
    identity = SimpleNamespace(model="M-02", sample="007", serial="SN-008")

    def start(request, lifecycle):
        requests.append(request)
        callbacks.append(lifecycle)
        assert states[-1][0] is False
        if outcome.on_start is not None:
            outcome.on_start()
        if outcome.error is not None:
            raise outcome.error
        return SimpleNamespace(request=request)

    def submit_archive(completion, product_name, recorded_on, callback):
        archive_submissions.append((completion, product_name, recorded_on, callback))
        if outcome.on_archive is None:
            callback(completion.request_id, None)
        else:
            outcome.on_archive(completion, callback)

    controller = MotorWorkbenchController(
        workbench.repository, workbench.panel, workbench.grid, workbench.products.append,
        lambda: None,
    )
    controller.bind_recording(WorkbenchRecordingPorts(
        bridge=SimpleNamespace(start=start, service=service),
        device=device, hardware_available=lambda snapshot: outcome.hardware_available,
        read_wav_identity=lambda: (identity.model, identity.sample, identity.serial),
        set_button_state=lambda *state: states.append(state),
        notify_error=problems.append, completed=completions.append,
        notify_cleanup_error=cleanup_problems.append,
        submit_archive=submit_archive, notify_archive_error=archive_problems.append,
    ))
    controller.refresh_configuration()

    def complete(index=-1, *, outcome="success", payload_id=None):
        request = requests[index]
        lifecycle = callbacks[index]
        session = SimpleNamespace(request=request)
        request_id = payload_id or request.request_id
        if outcome == "failure":
            lifecycle.failed(session, RecordingFailure(request_id, "capture", request.path, "设备断开"))
        elif outcome == "cancelled":
            lifecycle.cancelled(session, RecordingCancelled(request_id, request.path, 0, 0))
        else:
            group = request.station_group
            lifecycle.workbench_completed(session, WorkbenchRecordingCompletion(
                request_id, group.project_id, group.condition_id, request.sample_rate,
                request.target_samples - request.trim_samples, tuple(
                    StationWav(station.station_id, str(Path(group.final_path) / station.filename), station.channels)
                    for station in group.stations
                ),
            ))

    return SimpleNamespace(
        controller=controller, repository=workbench.repository, config=config, requests=requests,
        identity=identity,
        device=device, service=service, outcome=outcome, states=states, problems=problems,
        completions=completions, complete=complete, callbacks=callbacks,
        cleanup_problems=cleanup_problems, archive_problems=archive_problems,
        archive_submissions=archive_submissions,
    )


def test_recording_stays_busy_until_current_archive_result_is_delivered_once(recording_workbench):
    w = recording_workbench
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    w.outcome.on_archive = lambda *_: None
    w.controller.start_recording()
    assert w.states[-1][0::2] == (False, True)
    request, = w.requests
    w.complete()
    w.complete()
    w.complete(outcome="failure")
    w.complete(outcome="cancelled")
    (result, _, _, archive_finished), = w.archive_submissions
    archive_finished("older-request", None)
    w.callbacks[0].released(SimpleNamespace(request=request))
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests == [request]
    assert w.states[-1][0::2] == (False, True)
    assert "入库" in w.states[-1][1]
    assert w.completions == inputs == w.problems == []

    w.service.can_start_recording = False
    archive_finished(request.request_id, None)
    assert w.states[-1][0::2] == (False, False)
    assert w.completions == [result]
    assert [value.completion for value in inputs] == [result]
    w.service.can_start_recording = True
    w.callbacks[0].released(SimpleNamespace(request=request))
    assert w.states[-1][0::2] == (True, False)
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "cold"
    w.complete()
    archive_finished(request.request_id, None)
    assert w.completions == [result]
    assert w.states[-1][0::2] == (False, True)


def test_archive_and_analysis_keep_start_snapshot_across_midnight_and_product_switch(recording_workbench, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    w = recording_workbench
    clock = SimpleNamespace(now=lambda: datetime(2026, 11, 9, 23, 59, 59))
    monkeypatch.setattr(module, "datetime", clock)
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    w.outcome.on_archive = lambda *_: None
    w.controller.start_recording()

    clock.now = lambda: datetime(2026, 11, 10, 0, 0, 1)
    w.identity.model, w.identity.sample, w.identity.serial = "M-03", "009", "SN-010"
    w.repository.result = configuration_with(w.config, project_id="new-product", project_name="新产品")
    w.controller.refresh_configuration()
    w.complete()
    (result, product_name, recorded_on, archive_finished), = w.archive_submissions
    assert product_name == w.config.recording.context.project_name
    assert recorded_on.isoformat() == "2026-11-09"
    assert Path(result.stations[0].path).name == "2026-11-09_23-59-59-000_右工位_热机_M-02_007_SN-008.wav"
    archive_finished(result.request_id, None)
    assert w.completions == [result]
    assert inputs[0].context.analysis is w.config.analysis
    w.controller.start_recording()
    assert (w.requests[-1].station_group.project_id, w.requests[-1].station_group.condition_id) == (
        "new-product", "warm",
    )
    assert w.requests[-1].station_group.stations[0].filename == "2026-11-10_00-00-01-000_右工位_热机_M-03_009_SN-010.wav"


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_unpublished_recording_bypasses_archive_and_retries_condition(recording_workbench, outcome):
    w = recording_workbench
    w.controller.start_recording()
    w.complete(outcome=outcome)
    assert w.archive_submissions == w.completions == w.archive_problems == []
    assert w.states[-1][0::2] == (True, False)
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "warm"


@pytest.mark.parametrize("submit_fails", [False, True])
def test_archive_failure_delivers_recording_and_advances_condition(recording_workbench, caplog, submit_fails):
    w = recording_workbench
    inputs = []
    w.controller.bind_analysis_input(inputs.append)
    error = WorkbenchRecordingArchiveError("database\nlocked")

    def fail_archive(completion, callback):
        if submit_fails:
            raise error
        callback(completion.request_id, error)

    w.outcome.on_archive = fail_archive
    w.controller.start_recording()
    w.complete()
    w.complete()
    result, = w.completions
    assert [value.completion for value in inputs] == [result]
    assert w.archive_problems == ["工况「热机」音频已保存，入库失败：database locked"]
    assert not w.problems
    warning, = [record for record in caplog.records if record.levelname == "WARNING"]
    assert result.request_id in warning.getMessage()
    assert [station.path for station in result.stations] == warning.args[1]
    assert warning.args[2] is error
    if submit_fails:
        assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)
    assert w.states[-1][0::2] == (True, False)
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "cold"


def test_recording_retries_failures_and_wraps_in_declared_order(recording_workbench):
    w = recording_workbench
    start = w.controller.start_recording
    w.outcome.on_start = start
    w.outcome.error = RuntimeError("service became busy")
    assert w.states[-1][0] is True
    start()
    assert len(w.requests) == 1
    assert w.states[-1][0] is True
    assert "service became busy" in w.problems[-1]
    w.outcome.error = None
    start()
    start()
    w.complete(outcome="failure", payload_id="wrong")
    assert w.states[-1][0] is False
    w.complete(outcome="failure")
    assert "热机" in w.problems[-1] and "设备断开" in w.problems[-1]
    start()
    w.complete(outcome="cancelled")
    assert not w.completions
    start()
    w.complete()
    w.complete()
    start()
    w.complete(-2)
    assert w.states[-1][0] is False
    w.complete()
    start()
    assert [r.station_group.condition_id for r in w.requests] == ["warm"] * 4 + ["cold", "warm"]
    assert len({r.request_id for r in w.requests}) == len(w.requests)
    assert [r.condition_id for r in w.completions] == ["warm", "cold"]


def test_old_completion_is_delivered_without_advancing_new_configuration(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    second = configuration_with(w.config, project_id="other")
    w.repository.result = second
    w.controller.refresh_configuration()
    w.complete()
    assert [(r.project_id, r.condition_id) for r in w.completions] == [("product", "warm")]
    w.controller.start_recording()
    assert (w.requests[-1].station_group.project_id, w.requests[-1].station_group.condition_id) == (
        "other", "warm",
    )
    w.complete()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "cold"
    w.repository.result = w.config
    w.controller.refresh_configuration()
    w.repository.result = second
    w.controller.refresh_configuration()
    w.complete()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "warm"


@pytest.mark.parametrize("changed", ["root", "product", "condition", "station"])
def test_applied_output_changes_freeze_active_paths_and_restart_cursor(recording_workbench, tmp_path, changed, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: datetime(2026, 11, 9)))
    w = recording_workbench
    w.controller.start_recording()
    original = w.requests[0]
    root = w.config.recording.context.result_root_directory
    product = w.config.recording.context.project_name
    conditions = w.config.recording.context.conditions
    stations = w.config.recording.context.stations
    if changed == "root":
        root = str(tmp_path / "new-results")
    elif changed == "product":
        product = "新产品"
    elif changed == "condition":
        conditions = (replace(conditions[0], condition_name="新工况"), *conditions[1:])
    else:
        stations = (replace(stations[0], station_name="新工位"), *stations[1:])
    w.repository.result = configuration_with(w.config, project_name=product, result_root_directory=root,
                                  conditions=conditions, stations=stations)
    w.controller.refresh_configuration()
    w.complete()
    assert w.requests[0] is original
    assert Path(w.completions[0].stations[0].path).name == "2026-11-09_00-00-00-000_右工位_热机_M-02_007_SN-008.wav"
    assert Path(original.station_group.final_path) == Path(w.config.recording.context.result_root_directory) / "电机" / "audio"
    w.controller.start_recording()
    current = w.requests[-1]
    assert current.station_group.condition_id == "warm"
    assert Path(current.station_group.final_path) == Path(root) / product / "audio"
    assert current.station_group.stations[0].filename == (
        f"2026-11-09_00-00-00-000_{stations[0].station_name}_{conditions[0].condition_name}_M-02_007_SN-008.wav"
    )


def test_recording_timestamp_is_captured_before_request_preparation(recording_workbench, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    instant = datetime(2026, 11, 9, 13, 14, 15, 123999)
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: instant), raising=False)
    build = module.build_recording_request

    def delayed_build(*args, **kwargs):
        assert kwargs["recorded_at"] == instant
        assert (kwargs["product_model"], kwargs["sample_number"], kwargs["serial_number"]) == ("M-02", "007", "SN-008")
        return build(*args, **kwargs)

    monkeypatch.setattr(module, "build_recording_request", delayed_build)
    recording_workbench.controller.start_recording()
    assert recording_workbench.requests[0].station_group.stations[0].filename == (
        "2026-11-09_13-14-15-123_右工位_热机_M-02_007_SN-008.wav"
    )


def test_workbench_rejects_empty_sample_before_submitting_recording(recording_workbench):
    w = recording_workbench
    assert w.states[-1][0::2] == (True, False)
    w.identity.sample = "  "
    w.controller.start_recording()
    assert not w.requests
    assert "样本编号" in w.problems[-1]
    assert w.states[-1][0::2] == (True, False)

    w.identity.sample = "007"
    w.controller.start_recording()
    assert w.requests[0].station_group.condition_id == "warm"


def test_workbench_rejects_overlong_wav_name_before_submitting_recording(recording_workbench):
    w = recording_workbench
    w.identity.sample = "0" * 255
    w.controller.start_recording()
    assert not w.requests
    assert "右工位" in w.problems[-1] and "WAV filename length" in w.problems[-1]
    assert w.states[-1][0::2] == (True, False)


def test_invalid_product_root_notifies_ui_without_submission(recording_workbench):
    w = recording_workbench
    w.repository.result = configuration_with(w.config, result_root_directory="")
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert not w.requests
    assert "根目录" in w.problems[-1]


def test_equivalent_refresh_and_recovery_preserve_recording_progress(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    w.controller.refresh_configuration()
    condition, second = w.config.recording.context.conditions

    def with_detail(**changes):
        return dict(w.config.recording.parameters_by_queue) | {
            condition.test_queue: dict(w.config.recording.parameters_for(condition.condition_id)) | changes,
        }

    layout = w.config.ui.layout
    changed_cells = ((replace(layout.cells[0][0], title="新标题"), *layout.cells[0][1:]),)
    equivalent = configuration_with(
        w.config, plot_layout=replace(layout, cells=changed_cells),
        parameters=with_detail(startup_trim_ms=100.0, audio_validation={"unused_threshold": 2}),
        conditions=(replace(condition, applicable_station_ids=tuple(reversed(condition.applicable_station_ids))), second),
    )
    w.repository.result = equivalent
    w.controller.refresh_configuration()
    w.repository.result = WorkbenchConfigurationError("configuration unavailable")
    w.controller.refresh_configuration()
    assert not w.states[-1][0]
    w.complete()
    assert not w.states[-1][0]
    w.controller.start_recording()
    assert len(w.requests) == 1
    w.repository.result = equivalent
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "cold"
    w.repository.result = configuration_with(w.config, parameters=with_detail(total_time=3))
    w.controller.refresh_configuration()
    w.complete()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "warm"
    assert w.requests[-1].target_samples - w.requests[-1].trim_samples == 3 * 48000


@pytest.mark.parametrize("available", [None, ()])
def test_no_active_product_or_empty_conditions_disables_recording(recording_workbench, available):
    w = recording_workbench
    w.repository.result = None if available is None else configuration_with(w.config, conditions=available)
    w.controller.refresh_configuration()
    assert not w.states[-1][0]
    assert w.states[-1][1]
    w.controller.start_recording()
    assert not w.requests


def test_no_target_station_reports_specific_condition_without_submitting(recording_workbench):
    w = recording_workbench
    condition = replace(w.config.recording.context.conditions[0], applicable_station_ids=())
    w.repository.result = configuration_with(w.config, conditions=(condition,))
    w.controller.refresh_configuration()
    assert w.states[-1][0] is True
    w.controller.start_recording()
    assert not w.requests
    assert "热机" in w.problems[-1] and "工位" in w.problems[-1]
    assert w.states[-1][0] is True


def test_recording_admission_is_rechecked_on_each_click(recording_workbench):
    w = recording_workbench
    w.service.can_start_recording = False
    w.controller.start_recording()
    assert not w.states[-1][0]
    assert not w.requests
    w.service.can_start_recording = True
    w.outcome.hardware_available = False
    w.controller.start_recording()
    assert not w.states[-1][0]
    assert not w.requests
    w.outcome.hardware_available = True
    w.controller.refresh_recording_state()
    assert w.states[-1][0] is True


def test_device_generation_must_still_match_when_submitting(recording_workbench, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    w = recording_workbench
    build = module.build_recording_request
    snapshot = w.device.snapshot

    def build_while_device_is_invalidated(*args, **kwargs):
        request = build(*args, **kwargs)
        w.device.snapshot = None
        return request

    monkeypatch.setattr(module, "build_recording_request", build_while_device_is_invalidated)
    w.controller.start_recording()
    assert not w.requests
    assert not w.states[-1][0]
    monkeypatch.setattr(module, "build_recording_request", build)
    w.device.snapshot = replace(snapshot, generation=2)
    w.controller.refresh_recording_state()
    assert w.states[-1][0]
    w.controller.start_recording()
    assert w.requests[0].station_group.condition_id == "warm"


def test_equivalent_enabled_quality_threshold_uses_gate_numeric_semantics(recording_workbench):
    w = recording_workbench
    condition, second = w.config.recording.context.conditions

    def with_threshold(min_peak):
        parameters = dict(w.config.recording.parameters_by_queue)
        parameters[condition.test_queue] = dict(w.config.recording.parameters_for(condition.condition_id)) | {
            "audio_validation": {"enabled": True, "min_peak": min_peak},
        }
        return configuration_with(w.config, parameters=parameters)


    w.repository.result = with_threshold("0.25")
    w.controller.refresh_configuration()
    w.controller.start_recording()
    w.complete()
    w.repository.result = with_threshold(0.25)
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "cold"


def test_switching_registered_product_file_with_same_id_resets_progress(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    w.complete()
    w.repository.source_identity = "second-product.json"
    w.repository.result = w.config
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "warm"


def test_release_failure_reports_cleanup_and_refreshes_without_advancing(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    w.service.can_start_recording = False
    w.complete(outcome="failure")
    assert not w.states[-1][0]
    request, = w.requests
    w.service.can_start_recording = True
    w.callbacks[0].release_failed(SimpleNamespace(request=request), "temporary WAV retained")
    assert w.states[-1][0]
    assert "热机" in w.cleanup_problems[-1] and "temporary WAV retained" in w.cleanup_problems[-1]
    assert w.completions == []
    w.controller.start_recording()
    assert w.requests[-1].station_group.condition_id == "warm"


def test_inflight_success_does_not_advance_after_quality_limit_becomes_invalid(recording_workbench):
    w = recording_workbench
    condition, second = w.config.recording.context.conditions

    def with_threshold(encoded):
        parameters = dict(w.config.recording.parameters_by_queue)
        parameters[condition.test_queue] = dict(w.config.recording.parameters_for(condition.condition_id)) | {
            "audio_validation": json.loads(encoded),
        }
        return configuration_with(w.config, parameters=parameters)


    valid = with_threshold('{"enabled": true, "min_peak": "-inf"}')
    w.repository.result = valid
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests[0].station_group.condition_id == "warm"

    w.repository.result = with_threshold('{"enabled": true, "min_peak": -1e400}')
    w.controller.refresh_configuration()
    w.complete()
    assert [result.condition_id for result in w.completions] == ["warm"]
    w.controller.start_recording()
    assert len(w.requests) == 1
    assert "audio_validation.min_peak" in w.problems[-1]

    w.repository.result = valid
    w.controller.refresh_configuration()
    w.controller.start_recording()
    assert w.requests[1].station_group.condition_id == "warm"


def bind_product_selection(workbench):
    from ui.sequence.motor_workbench_controller import WorkbenchProductSelectionPorts

    state = SimpleNamespace(busy=False, closing=False, saved=[], catalogs=[], enabled=[], errors=[], save_result=(True, ""))

    def save(file_name):
        workbench.repository.calls.append(("save", file_name))
        state.saved.append(file_name)
        if isinstance(state.save_result, Exception):
            raise state.save_result
        return state.save_result

    state.manager = SimpleNamespace(set_active_product=save)
    workbench.controller.bind_product_selection(WorkbenchProductSelectionPorts(
        product_manager=state.manager,
        show_catalog=lambda entries, selected: state.catalogs.append((entries, selected)),
        set_enabled=state.enabled.append,
        is_busy=lambda: state.busy,
        is_closing=lambda: state.closing,
        notify_error=state.errors.append,
    ))
    return state


@pytest.fixture
def selection_workbench(workbench, recording_workbench):
    workbench.controller = recording_workbench.controller
    workbench.products.clear()
    state = bind_product_selection(workbench)
    workbench.repository.candidate = configuration_with(recording_workbench.config, project_name="另一电机")
    workbench.controller.refresh_configuration()
    workbench.selection = state
    return workbench


def selection_snapshot(w):
    controller = w.controller
    return (
        controller.get_recording_config(), controller.get_analysis_config(), controller.get_ui_config(),
        controller._analysis_version, controller._configuration_revision, controller._condition_index,
        controller._configuration_valid, w.repository.source_identity, controller._recording_key,
        controller._recording_admission(),
        w.panel.get_station_data(1),
        tuple(plot_cells(w.grid)[0, 0].plot_widget.getPlotItem().listDataItems()),
    )


def populate_selection_results(w):
    w.panel.set_station_results(1, "OK", "NG")
    w.panel.set_station_cell(1, 0, 1, 12)
    w.grid.set_curve_data(0, 0, [{"x": [1, 2], "y": [3, 4]}])
    w.controller._condition_index = 1


def test_product_selection_catalog_refresh_keeps_applied_source(selection_workbench):
    w = selection_workbench
    assert w.selection.catalogs[-1] == (w.repository.catalog, "product.json")
    assert w.products == []
    w.repository.catalog = (("second.json", "同名"), ("product.json", "同名"), ("third.json", "新增"))
    w.controller.refresh_product_catalog()
    assert w.selection.catalogs[-1] == (w.repository.catalog, "product.json")
    assert w.selection.saved == []


@pytest.mark.parametrize("equal_content", [False, True])
def test_product_selection_prepares_once_before_save_and_resets_source(selection_workbench, monkeypatch, equal_content):
    from ui.sequence import motor_workbench_controller as module

    w = selection_workbench
    populate_selection_results(w)
    previous = selection_snapshot(w)
    if equal_content:
        w.repository.candidate = configuration_with(w.repository.result)
    candidate = w.repository.candidate
    projections = []
    build = module.build_ui_projection

    def prepare(ui, analysis):
        w.calls.append("prepare")
        projection = build(ui, analysis)
        projections.append(projection)
        return projection

    monkeypatch.setattr(module, "build_ui_projection", prepare)
    w.calls.clear()
    w.controller.select_product("second.json")
    assert w.calls == ["catalog", ("candidate", "second.json"), "prepare", ("save", "second.json"),
                       ("accept", "second.json"), "panel", "grid"]
    assert len(projections) == 1
    assert w.plot_batches[-1][0] is projections[0].column_names
    assert w.plot_batches[-1][1] is projections[0].cells
    assert w.controller.get_recording_config() is candidate.recording
    assert w.controller._plot_bindings is projections[0].bindings
    assert w.repository.source_identity == str(Path("products") / "second.json")
    assert w.controller._analysis_version != previous[3]
    assert w.controller._configuration_revision > previous[4]
    assert w.controller._condition_index == 0
    assert w.panel.get_station_data(1)["forward"] == "--"
    assert not plot_cells(w.grid)[0, 0].plot_widget.getPlotItem().listDataItems()
    assert w.selection.catalogs[-1] == (w.repository.catalog, "second.json")
    assert w.selection.enabled[-1] is True


def test_product_selection_same_source_preserves_everything(selection_workbench):
    w = selection_workbench
    populate_selection_results(w)
    previous = selection_snapshot(w)
    w.calls.clear()
    w.controller.select_product("product.json")
    assert selection_snapshot(w) == previous
    assert w.calls == ["catalog"]
    assert w.selection.saved == []
    assert w.selection.catalogs[-1][1] == "product.json"


@pytest.mark.parametrize("failure", ["candidate", "projection", "save", "io", "removed", "catalog"])
def test_product_selection_expected_failure_preserves_snapshot(selection_workbench, monkeypatch, failure):
    from ui.sequence import motor_workbench_controller as module

    w = selection_workbench
    populate_selection_results(w)
    previous = selection_snapshot(w)
    if failure == "candidate":
        w.repository.candidate = WorkbenchConfigurationError("second.json: invalid queue")
    elif failure == "projection":
        def fail_projection(*_):
            raise WorkbenchConfigurationError("layout.json: invalid projection")
        monkeypatch.setattr(module, "build_ui_projection", fail_projection)
    elif failure == "save":
        w.selection.save_result = (False, "registry.json: write failed")
    elif failure == "io":
        w.selection.save_result = OSError("registry.json: permission denied")
    elif failure == "removed":
        w.repository.catalog = (("product.json", "电机"),)
    else:
        w.repository.catalog = WorkbenchConfigurationError("registry.json: invalid registry")
    w.calls.clear()
    w.controller.select_product("second.json")
    assert selection_snapshot(w) == previous
    assert "panel" not in w.calls and "grid" not in w.calls
    assert not any(isinstance(call, tuple) and call[0] == "accept" for call in w.calls)
    assert len(w.selection.errors) == 1
    assert w.selection.catalogs[-1][1] == "product.json"
    assert w.selection.enabled[-1] is True


@pytest.mark.parametrize("closing", [False, True])
def test_product_selection_busy_entry_restores_before_reads(selection_workbench, closing):
    w = selection_workbench
    w.selection.closing = closing
    w.selection.busy = not closing
    w.calls.clear()
    w.controller.select_product("second.json")
    assert w.calls == []
    assert w.selection.saved == []
    assert w.selection.catalogs[-1][1] == "product.json"
    assert w.selection.enabled[-1] is False
    w.selection.closing = w.selection.busy = False
    w.controller.refresh_recording_state()
    assert w.selection.enabled[-1] is True


def test_product_selection_rechecks_busy_after_preparation(selection_workbench, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    w = selection_workbench
    previous = selection_snapshot(w)
    build = module.build_ui_projection

    def prepare(*args):
        projection = build(*args)
        w.selection.busy = True
        return projection

    monkeypatch.setattr(module, "build_ui_projection", prepare)
    w.controller.select_product("second.json")
    assert selection_snapshot(w) == previous
    assert w.selection.saved == []
    assert w.selection.enabled[-1] is False
    assert w.selection.catalogs[-1][1] == "product.json"


def test_product_selection_without_recording_refreshes_busy_state(workbench):
    selection = bind_product_selection(workbench)
    assert selection.catalogs[-1] == (workbench.repository.catalog, "")
    assert selection.enabled[-1] is True
    selection.busy = True
    workbench.controller.refresh_recording_state()
    assert selection.enabled[-1] is False
    selection.busy = False
    workbench.controller.refresh_recording_state()
    assert selection.enabled[-1] is True
    assert selection.saved == []


def test_product_selection_catalog_failure_retains_entries_and_recovers(selection_workbench, caplog):
    w = selection_workbench
    previous = w.selection.catalogs[-1]
    w.repository.catalog = WorkbenchConfigurationError("registry.json: unavailable")
    w.controller.refresh_product_catalog()
    assert w.selection.catalogs[-1] == previous
    assert "registry.json: unavailable" in caplog.text
    assert w.selection.errors == ["registry.json: unavailable"]
    w.repository.catalog = (("product.json", "新名称"),)
    w.controller.refresh_product_catalog()
    assert w.selection.catalogs[-1] == (w.repository.catalog, "product.json")
    assert w.selection.saved == []


def test_product_selection_external_refresh_tracks_source_and_equal_notifications(selection_workbench):
    w = selection_workbench
    populate_selection_results(w)
    previous_version = w.controller._analysis_version
    w.repository.source_identity = str(Path("products") / "second.json")
    w.controller.refresh_configuration()
    assert w.controller._analysis_version != previous_version
    assert w.controller._condition_index == 0
    assert w.panel.get_station_data(1)["forward"] == "--"
    assert w.selection.catalogs[-1][1] == "second.json"
    populate_selection_results(w)
    previous = selection_snapshot(w)
    w.controller.refresh_configuration()
    assert selection_snapshot(w) == previous
    assert w.selection.saved == []


def test_product_selection_editor_refresh_during_recording_keeps_attempt(recording_workbench):
    w = recording_workbench
    selection = bind_product_selection(w)
    w.controller.start_recording()
    attempt = w.controller._in_flight
    w.repository.result = configuration_with(w.config, project_name="编辑后的产品")
    w.repository.source_identity = str(Path("products") / "second.json")
    w.controller.refresh_configuration()
    assert w.controller._in_flight is attempt
    assert w.controller.get_recording_config() is w.repository.result.recording
    assert w.controller._analysis_version != attempt.analysis_context.analysis_version
    assert selection.catalogs[-1][1] == "second.json"
    assert selection.enabled[-1] is False
    assert selection.saved == []
    w.complete()
    assert w.controller._condition_index == 0
    assert selection.enabled[-1] is True


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled", "rejected"])
def test_product_selection_recording_archive_cleanup_and_analysis_recovery(recording_workbench, terminal):
    w = recording_workbench
    selection = bind_product_selection(w)
    w.repository.candidate = w.config
    w.outcome.on_archive = lambda *_: None
    if terminal == "rejected":
        w.outcome.error = RuntimeError("service busy")
    w.controller.start_recording()
    if terminal == "rejected":
        assert selection.enabled[-1] is True
        return
    assert selection.enabled[-1] is False
    w.repository.calls.clear()
    w.controller.select_product("second.json")
    assert w.repository.calls == []
    w.service.can_start_recording = False
    selection.busy = True  # recording resources are still being released
    w.complete(outcome=terminal)
    assert selection.enabled[-1] is False
    if terminal == "success":
        assert w.controller._in_flight.archiving
        w.repository.calls.clear()
        w.controller.select_product("second.json")
        assert w.repository.calls == []
        result, _, _, archived = w.archive_submissions[-1]
        archived(result.request_id, None)
        assert selection.enabled[-1] is False
    w.service.can_start_recording = True
    selection.busy = False
    w.callbacks[-1].released(SimpleNamespace(request=w.requests[-1]))
    assert selection.enabled[-1] is True
    for phase in ("queued analysis", "active analysis"):
        selection.busy = True
        w.controller.refresh_recording_state()
        assert selection.enabled[-1] is False, phase
        selection.busy = False
        w.controller.refresh_recording_state()
        assert selection.enabled[-1] is True, phase


def test_product_selection_unexpected_view_error_propagates_with_source(selection_workbench, monkeypatch, caplog):
    w = selection_workbench
    caplog.set_level("INFO", logger="core")

    def fail(*_):
        raise RuntimeError("view publication failed")

    monkeypatch.setattr(w.panel, "show_workbench_stations", fail)
    with pytest.raises(RuntimeError, match="view publication failed"):
        w.controller.select_product("second.json")
    assert w.selection.saved == ["second.json"]
    assert "second.json" in caplog.text
    assert w.selection.errors == []
