import json
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

import pytest

from base.workbench_config_values import (
    WorkbenchChannel, WorkbenchCondition, WorkbenchContext,
    WorkbenchHardwareChannel, WorkbenchStation,
    _same_snapshot_value, _thaw_json,
)
from base.workbench_recording_config import WorkbenchRecordingConfig
from base.workbench_analysis_config import AnalysisTestQueue, WorkbenchAnalysisConfig
from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchPlotLayout, WorkbenchUiConfig
from unit_test.base.workbench_config_fixtures import workbench_source


def make_values(source):
    product = source["product"]
    stations = [WorkbenchStation(
        station["station_id"], station["station_name"], [WorkbenchChannel(
            channel["channel_id"], channel["channel_name"], channel["signal_type"],
            WorkbenchHardwareChannel(**channel["hardware_channel"]),
        ) for channel in station["channels"]],
    ) for station in product["stations"]]
    context = WorkbenchContext(
        product["project_id"], product["project_name"], product["result_root_directory"],
        stations, [WorkbenchCondition(**item) for item in product["conditions"]],
    )
    recording = WorkbenchRecordingConfig(context, {name: {
        key: seq[0]["seq1"]["acq"]["detail"][key]
        for key in ("total_time", "startup_trim_ms", "audio_validation")
    } for name, seq in source["queues"].items()})
    analysis = WorkbenchAnalysisConfig(context, {name: AnalysisTestQueue(
        seq[0]["seq1"].get("analysis_channel_basis"), seq[0]["seq1"]["analysis_list"],
    ) for name, seq in source["queues"].items()})
    layout = source["layout"]
    ui = WorkbenchUiConfig(WorkbenchPlotLayout(
        layout["rows"], layout["columns"],
        [[WorkbenchPlotCell(**cell) for cell in row] for row in layout["cells"]],
    ), {name: "cumulative" for name in source["queues"]})
    return recording, analysis, ui


@pytest.mark.parametrize("all_stations, expected", [
    (False, (("s1", "s2"), ("s2", "s4"))),
    (True, (("s1", "s2", "s3", "s4"),) * 2),
])
def test_shared_context_queries_and_recording_parameters(all_stations, expected):
    recording, analysis, _ = make_values(workbench_source(all_stations=all_stations))
    assert recording.context is analysis.context
    assert analysis.context.get_condition("c1") is analysis.context.conditions[0]
    assert analysis.context.get_station("s1") is analysis.context.stations[0]
    assert analysis.context.get_condition("unknown") is None
    assert analysis.context.get_station("unknown") is None
    for index, station_ids in enumerate(expected):
        assert tuple(s.station_id for s in analysis.stations_for(f"c{index + 1}")) == station_ids
    assert analysis.stations_for("c1")[0] is analysis.context.stations[0]
    assert recording.parameters_for("c1")["total_time"] == 2.5
    assert recording.parameters_for("c2")["total_time"] == 3.5
    for lookup in (recording.parameters_for, analysis.stations_for):
        with pytest.raises(ValueError, match="unknown"):
            lookup("unknown")


def test_describe_preserves_complete_contract_and_independent_exports():
    source = workbench_source()
    recording, analysis, _ = make_values(source)
    before = analysis.describe()
    after = analysis.describe()
    assert set(after) == {"stations", "conditions", "test_queues", "tasks"}
    assert after["tasks"] == []
    assert [s["station_id"] for s in after["stations"]] == ["s1", "s2", "s4"]
    assert after["stations"][1] == {
        "station_id": "s2", "station_name": "工位2", "device_ref": "device-1",
        "channels": [{"channel_id": f"ch{i + 1}", "channel_name": f"声音{i + 1}",
                      "signal_type": "audio", "physical_channel_index": 2 + i} for i in range(2)],
    }
    assert after["conditions"][0] == {
        "condition_id": "c1", "condition_name": "工况1", "test_queue": "测试队列A",
        "segmented_analysis": source["product"]["conditions"][0]["segmented_analysis"],
    }
    for name, sequence in source["queues"].items():
        assert after["test_queues"][name] == {
            "analysis_channel_basis": "wav", "analysis_config": sequence[0]["seq1"]["analysis_list"],
        }
    before["test_queues"]["测试队列A"]["analysis_config"]["声压级"]["analysis_channels"].append(99)
    before["conditions"][0]["segmented_analysis"]["settings"]["intervals"][0].append(99)
    before["stations"][0]["channels"][0]["physical_channel_index"] = 99
    assert after == analysis.describe()
    assert after["test_queues"]["测试队列A"]["analysis_config"]["声压级"]["analysis_channels"] == [0]
    assert json.loads(json.dumps(after)) == after
    assert recording.parameters_for("c1") is recording.parameters_by_queue["测试队列A"]


def test_absent_basis_and_unassigned_or_empty_stations_are_describable():
    source = workbench_source()
    del source["queues"]["测试队列A"][0]["seq1"]["analysis_channel_basis"]
    source["product"]["conditions"][0]["applicable_station_ids"] = []
    source["product"]["stations"][1]["channels"] = []
    _, analysis, _ = make_values(source)
    assert analysis.test_queues["测试队列A"].analysis_channel_basis is None
    assert analysis.stations_for("c1") == ()
    assert analysis.stations_for("c2")[0].channels == ()
    assert [s["station_id"] for s in analysis.describe()["stations"]] == ["s4"]
    assert len(analysis.describe()["conditions"]) == 2
    assert analysis.describe()["test_queues"]["测试队列A"]["analysis_channel_basis"] is None


def test_construction_freezes_source_containers_and_public_values():
    source = workbench_source()
    recording, analysis, ui = make_values(source)
    before = analysis.describe()
    source["product"]["conditions"][0]["applicable_station_ids"].clear()
    source["product"]["conditions"][0]["segmented_analysis"]["settings"]["intervals"][0].append(99)
    source["queues"]["测试队列A"][0]["seq1"]["analysis_list"]["声压级"]["limits"]["values"][0].append(99)
    source["queues"]["测试队列A"][0]["seq1"]["acq"]["detail"]["audio_validation"]["enabled"] = False
    assert analysis.describe() == before
    assert recording.parameters_for("c1")["audio_validation"]["enabled"] is True
    assert isinstance(analysis.context.stations, tuple)
    assert isinstance(analysis.context.conditions, tuple)
    assert isinstance(analysis.context.stations[0].channels, tuple)
    assert isinstance(ui.layout.cells, tuple)
    assert isinstance(ui.layout.cells[0], tuple)
    with pytest.raises(FrozenInstanceError):
        analysis.context.project_name = "changed"
    with pytest.raises(TypeError):
        analysis.test_queues["new"] = analysis.test_queues["测试队列A"]
    with pytest.raises(TypeError):
        analysis.test_queues["测试队列A"].analysis_config["声压级"]["limits"]["values"][0][0] = 99
    with pytest.raises(TypeError):
        analysis.context.conditions[0].segmented_analysis["settings"]["intervals"] = []
    with pytest.raises(TypeError):
        recording.parameters_for("c1")["audio_validation"]["enabled"] = False
    with pytest.raises(TypeError):
        ui.preview_time_modes["测试队列A"] = "other"


def test_mapping_proxy_and_tuple_inputs_are_copied_recursively():
    inner = {"values": [1, {"enabled": True}]}
    test_queue = AnalysisTestQueue(None, MappingProxyType({"custom": (inner,)}))
    test_queues = {"queue": test_queue}
    context = WorkbenchContext("p", "product", "root", [], [])
    analysis = WorkbenchAnalysisConfig(context, test_queues)
    inner["values"][1]["enabled"] = False
    test_queues.clear()
    assert analysis.describe()["test_queues"]["queue"]["analysis_config"] == {
        "custom": [{"values": [1, {"enabled": True}]}],
    }
    assert analysis.test_queues["queue"] is test_queue


def test_outer_sequence_inputs_and_display_map_are_detached():
    station = WorkbenchStation("s", "station", [])
    stations = [station]
    conditions = []
    context = WorkbenchContext("p", "product", "root", stations, conditions)
    cells = [[WorkbenchPlotCell("title", "metric", "SPL")]]
    modes = {"queue": "cumulative"}
    ui = WorkbenchUiConfig(WorkbenchPlotLayout(1, 1, cells), modes)
    stations.clear()
    conditions.append("unexpected")
    cells[0].clear()
    modes.clear()
    assert context.stations == (station,)
    assert context.conditions == ()
    assert ui.layout.cells[0][0].title == "title"
    assert ui.preview_time_modes["queue"] == "cumulative"


@pytest.mark.parametrize("part", ["analysis", "condition", "recording", "channel"])
def test_snapshot_equality_distinguishes_boolean_and_integer(part):
    first = workbench_source()
    second = workbench_source()
    if part == "analysis":
        first["queues"]["测试队列A"][0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] = [1]
        second["queues"]["测试队列A"][0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] = [True]
    elif part == "condition":
        second["product"]["conditions"][0]["segmented_analysis"]["settings"]["intervals"][0][1] = True
    elif part == "recording":
        second["queues"]["测试队列A"][0]["seq1"]["acq"]["detail"]["audio_validation"]["enabled"] = 1
    else:
        second["product"]["stations"][0]["channels"][0]["hardware_channel"]["physical_channel_index"] = False
    a_recording, a_analysis, a_ui = make_values(first)
    b_recording, b_analysis, b_ui = make_values(second)
    assert make_values(first) == (a_recording, a_analysis, a_ui)
    if part == "recording":
        assert a_recording != b_recording
        assert a_analysis == b_analysis
    else:
        assert a_analysis != b_analysis
    assert a_ui == b_ui


def test_values_have_only_their_owned_fields_and_share_same_queue_parameters():
    recording, analysis, ui = make_values(workbench_source())
    assert {f.name for f in fields(recording.context)} == {
        "project_id", "project_name", "result_root_directory", "stations", "conditions",
    }
    assert {f.name for f in fields(recording.context.conditions[0])} == {
        "condition_id", "condition_name", "test_queue", "applicable_station_ids", "trigger_state", "segmented_analysis",
    }
    assert set(vars(recording)) == {"context", "parameters_by_queue"}
    assert set(vars(analysis)) == {"context", "test_queues"}
    assert set(vars(ui)) == {"layout", "preview_time_modes"}
    shared = replace(recording.context.conditions[1], test_queue="测试队列A")
    same_queue = replace(recording, context=replace(recording.context, conditions=(recording.context.conditions[0], shared)))
    assert same_queue.parameters_for("c1") is same_queue.parameters_for("c2")


def test_deep_freezing_reports_field_and_thaw_and_comparison_are_iterative():
    nested = {"leaf": True}
    for _ in range(1500):
        nested = {"next": nested}
    with pytest.raises(ValueError, match="analysis_config.*nested too deeply"):
        AnalysisTestQueue(None, nested)
    with pytest.raises(ValueError, match="segmented_analysis.*nested too deeply"):
        WorkbenchCondition("c", "condition", "queue", [], "", nested)
    exported = _thaw_json(nested)
    assert exported is not nested
    assert _same_snapshot_value(nested, exported)
    current = exported
    for _ in range(1500):
        current = current["next"]
    current["leaf"] = 1
    assert not _same_snapshot_value(nested, exported)
