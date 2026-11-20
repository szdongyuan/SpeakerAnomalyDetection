"""Configured condition columns and per-station verdict/terminal boundaries."""

from dataclasses import replace
from datetime import datetime

import pytest
from PyQt5.QtWidgets import QTableWidget

from ui.sequence.motor_station_test_panel import MotorStationTestPanel
from ui.sequence.workbench_result_state import WorkbenchResultState
from unit_test.ui.test_workbench_result_state import instance, payload
from unit_test.ui.test_workbench_result_sync import (
    sync, config, workbench, recording_workbench, paired_recording_workbench,
    configuration_with, early_event, result_event, labels,
)


def test_three_conditions_affect_early_verdict_terminal_and_daily_summary(sync):
    conditions = sync.w.config.analysis.context.conditions
    third = replace(conditions[0], condition_id="load", condition_name="负载测试")
    sync.w.repository.result = configuration_with(sync.w.config, conditions=(*conditions, third))
    sync.w.controller.refresh_configuration()
    tasks = [sync.record()["right"] for _ in range(3)]
    table = sync.panel.findChild(QTableWidget, "stationTable1")
    assert [table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "指标", *(c.condition_name for c in (*conditions, third))]
    # Third condition arrives first; neither missing results nor display order may be ignored.
    sync.deliver(early_event(tasks[2], "NG"), early_event(tasks[0]))
    assert sync.panel.get_station_data(1)["result"] == "--"
    sync.deliver(early_event(tasks[1]))
    assert sync.panel.get_station_data(1)["result"] == "NG"
    assert labels(sync.summary)[0] == "0"
    for index in (1, 2, 0):
        verdict = "NG" if index == 2 else "OK"
        sync.deliver(result_event(tasks[index], [instance(judgement=verdict)]))
    assert labels(sync.summary)[:4] == ("1", "0", "1", "0")
    sync.deliver(result_event(tasks[2], [instance(judgement="NG")]))
    assert labels(sync.summary)[0] == "1"


def test_each_station_completes_only_its_applicable_conditions(sync):
    conditions = sync.w.config.analysis.context.conditions
    selected = (replace(conditions[0], applicable_station_ids=("right",)),
                replace(conditions[1], applicable_station_ids=("left",)))
    sync.w.repository.result = configuration_with(sync.w.config, conditions=selected)
    sync.w.controller.refresh_configuration()
    first, second = sync.record(), sync.record()
    assert set(first) == {"right"} and set(second) == {"left"}
    for task in (first["right"], second["left"]):
        sync.deliver(early_event(task))
    assert all(sync.panel.get_station_data(i)["result"] == "OK" for i in (1, 2))
    assert labels(sync.summary)[0] == "0"
    for task in (first["right"], second["left"]):
        sync.deliver(result_event(task, [instance()]))
    assert labels(sync.summary)[:4] == ("2", "2", "0", "0")
    assert sync.panel.get_station_data(1)["reverse"] == "--"
    assert sync.panel.get_station_data(2)["forward"] == "--"


def test_single_condition_has_one_value_column_and_completes(sync):
    condition = sync.w.config.analysis.context.conditions[0]
    sync.w.repository.result = configuration_with(sync.w.config, conditions=(condition,))
    sync.w.controller.refresh_configuration()
    task = sync.record()["right"]
    sync.deliver(early_event(task))
    assert sync.panel.get_station_data(1)["result"] == "OK"
    assert labels(sync.summary)[0] == "0"
    sync.deliver(result_event(task, [instance()]))
    assert labels(sync.summary)[:4] == ("1", "1", "0", "0")
    assert len(sync.panel.get_station_data(1)["rows"][0]) == 2


@pytest.mark.parametrize("count", [0, 1, 2, 3, 6])
def test_dynamic_columns_refresh_and_scroll(qt_app, tmp_path, count):
    panel = MotorStationTestPanel()
    panel.resize(340, 360)
    names = tuple(f"工况{i + 1}" for i in range(count))
    panel.show_workbench_stations((("工位1", ("声压级",)),), condition_names=names)
    panel.show_condition_result(1, ("OK",) * count, "OK" if count else "--",
                                [["声压级", *(str(60 + i) for i in range(count))]])
    panel._stations[0].header.setChecked(True)
    panel.show()
    qt_app.processEvents()
    table = panel.findChild(QTableWidget, "stationTable1")
    assert table.columnCount() == count + 1
    assert [table.horizontalHeaderItem(i).text() for i in range(count + 1)] == ["指标", *names]
    assert [label.text() for label in panel._general_header._labels] == ["工位", *names, "结果"]
    bar = panel._scroll.horizontalScrollBar()
    assert (bar.maximum() > 0) == (count > 2)
    bar.setValue(bar.maximum())
    qt_app.processEvents()
    assert panel._general_header.x() == -bar.value()
    for heading, value in zip(panel._general_header._labels, panel._stations[0].header._labels):
        assert abs(heading.mapTo(panel, heading.rect().center()).x()
                   - value.mapTo(panel, value.rect().center()).x()) <= 1
    panel.grab().save(str(tmp_path / f"conditions_{count}.png"))
    panel.show_workbench_stations((("工位1", ("声压级",)),), condition_names=("复检",))
    assert panel.findChild(QTableWidget, "stationTable1").columnCount() == 2
    assert panel.get_station_data(1)["result"] == "--"
    panel.close()
    panel.deleteLater()


def test_unexpected_station_condition_is_rejected():
    state = WorkbenchResultState()
    state.open_round("r", analysis_version="v", source_identity="review", recorded_at=datetime.now(),
                     condition_ids=("a", "b"), stations=(("station-a", 1, ("SPL",)),),
                     product_code="p", expected_analysis_keys=frozenset({("a", "station-a")}))
    with pytest.raises(ValueError, match="Unexpected condition"):
        state.receive("r", "station-a", "b", payload("b", [instance()]))
    result = state.receive("r", "station-a", "a", payload("a", [instance()]))
    assert result.complete and result.result == "OK"
    assert result.condition_statuses == ("OK", "--")


def test_third_required_failure_cannot_be_hidden_by_first_two_ok():
    state = WorkbenchResultState()
    state.open_round("r", analysis_version="v", source_identity="review", recorded_at=datetime.now(),
                     condition_ids=("a", "b", "c"), stations=(("station-a", 1, ()),), product_code="p")
    for condition in ("a", "b"):
        state.receive("r", "station-a", condition, payload(condition, [instance()]))
    failed = payload("c", [])
    failed["judgement_summary"] = {"status": "结果不完整"}
    result = state.receive("r", "station-a", "c", failed)
    assert result.complete and result.result == "结果不完整"
