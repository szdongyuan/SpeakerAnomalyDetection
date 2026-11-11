from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import pytest

from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_configuration_loader import load_workbench_configuration
from base.workbench_ui_config import WorkbenchPlotCell, WorkbenchPlotLayout
from unit_test.base.workbench_config_fixtures import write_workbench_files
from ui.sequence.workbench_ui_projection import build_ui_projection


@pytest.fixture
def configuration(tmp_path):
    files = write_workbench_files(tmp_path)
    return load_workbench_configuration(files["programs"], files["queues"], files["layout"])


def test_projection_preserves_layout_partial_associations_and_conflicting_units(configuration):
    projection = build_ui_projection(configuration.ui, configuration.analysis)

    assert projection.stations == (
        ("工位1", ("声压级", "频谱")),
        ("工位2", ("声压级（工况1）", "频谱（工况1）", "声压级（工况2）", "频谱（工况2）")),
        ("工位3", ()),
        ("工位4", ("声压级", "频谱")),
    )
    assert projection.column_names == ("工位1", "工位2", "工位3", "工位4")
    assert len(projection.cells) == len(projection.bindings) == 2
    assert all(len(row) == 4 for row in projection.cells + projection.bindings)
    cell = projection.cells[0][1]
    assert (cell.title, cell.unit, cell.first_condition, cell.second_condition) == (
        "声压级2", "dB(A)", "工况1", "",
    )
    assert cell.warning == "单位不一致"
    assert cell.warning_detail.splitlines() == ["工况1 dB(A)", "工况2 dB(C)"]
    first, second = projection.bindings[0][1]
    assert (first.station_id, first.condition_id, first.analysis_item, first.analysis_type) == (
        "s2", "c1", "声压级", "SPL",
    )
    assert second is None
    assert projection.cells[0][2].title == "声压级3"
    assert projection.bindings[0][2] == (None, None)
    assert projection.bindings[0][3][0].condition_id == "c2"
    assert projection.cells[1][1].unit == "dB(Z)"
    assert tuple(binding.condition_id for binding in projection.bindings[1][1]) == ("c1", "c2")
    assert projection.preview_time_modes == {"c1": "cumulative", "c2": "cumulative"}
    with pytest.raises(FrozenInstanceError):
        projection.column_names = ()
    with pytest.raises(TypeError):
        projection.preview_time_modes["c1"] = "window"


def test_absent_layout_projects_no_plots_but_keeps_stations_and_preview_modes(configuration):
    expected = build_ui_projection(configuration.ui, configuration.analysis)
    ui = replace(configuration.ui, layout=None)

    projection = build_ui_projection(ui, configuration.analysis)

    assert projection.stations == expected.stations
    assert projection.preview_time_modes == expected.preview_time_modes
    assert projection.column_names == projection.cells == projection.bindings == ()


def test_missing_items_and_types_keep_condition_slots_and_custom_titles(configuration):
    # The first queue has an FFT item that is deliberately not in display order.
    test_queues = dict(configuration.analysis.test_queues)
    first = test_queues["测试队列A"]
    test_queues["测试队列A"] = replace(first, analysis_config={
        **first.analysis_config, "display_sequence": ("声压级",),
    })
    analysis = replace(configuration.analysis, test_queues=test_queues)
    cell = WorkbenchPlotCell(" 原文标题 <FFT> ", "频谱", "FFT")
    wrong_type = WorkbenchPlotCell("类型错误", "声压级", "FFT")
    ui = replace(configuration.ui, layout=WorkbenchPlotLayout(2, 5, ((cell,) * 5, (wrong_type,) * 5)))
    projection = build_ui_projection(ui, analysis)

    display = projection.cells[0][1]
    assert (display.title, display.unit, display.first_condition, display.second_condition) == (
        cell.title, "dB(Z)", "工况1", "工况2",
    )
    assert display.warning == "分析项不可用"
    assert display.warning_detail == "工况1：缺少分析项「频谱」"
    assert projection.bindings[0][1][0] is None
    assert projection.bindings[0][1][1].condition_id == "c2"
    display = projection.cells[1][1]
    assert (display.first_condition, display.second_condition) == ("工况1", "工况2")
    assert display.warning == "分析项不可用"
    assert display.warning_detail.splitlines() == [
        "工况1：分析项「声压级」类型不符", "工况2：分析项「声压级」类型不符",
    ]
    assert projection.bindings[1][1] == (None, None)
    assert projection.column_names[-1] == ""
    assert projection.cells[0][4].title == ""
    assert projection.bindings[0][4] == (None, None)


@pytest.mark.parametrize("kind", ["SPL", "FFT"])
@pytest.mark.parametrize(("raw", "unit"), [
    ("a", "dB(A)"), ("c", "dB(C)"), ("None", "dB(Z)"),
    ("Z（NONE）", "dB(Z)"), (None, "dB(Z)"), ("B", "dB(B)"),
    ("custom-weighting", "dB(CUSTOM-WEIGHTING)"),
])
def test_units_are_display_only_and_preserve_full_analysis(configuration, kind, raw, unit):
    item = {"type": f" {kind} ", "baseline_display_mode": "delta", "custom": {"values": [1, 2]}}
    if raw is not None:
        item["weighting"] = raw
    analysis = replace(configuration.analysis, test_queues={
        name: AnalysisTestQueue("wav", {"display_sequence": ["任意分析项"], "任意分析项": item})
        for name in configuration.analysis.test_queues
    })
    cell = WorkbenchPlotCell("不改名称", "任意分析项", kind)
    ui = replace(configuration.ui, layout=WorkbenchPlotLayout(1, 4, ((cell,) * 4,)))
    before = analysis.describe()
    projection = build_ui_projection(ui, analysis)

    display = projection.cells[0][1]
    assert (display.unit, display.first_condition, display.second_condition) == (unit, "工况1", "工况2")
    assert display.warning == display.warning_detail == ""
    assert tuple(binding.condition_id for binding in projection.bindings[0][1]) == ("c1", "c2")
    assert analysis.describe() == before
    assert before["test_queues"]["测试队列A"]["analysis_config"]["任意分析项"] == item


def test_projection_contains_only_display_values_and_limits_plots_to_first_two_conditions(configuration):
    context = configuration.analysis.context
    third = replace(context.conditions[0], condition_id="c3", condition_name="工况3", applicable_station_ids=("s2",))
    unused = replace(third, condition_id="c4", condition_name="未关联", applicable_station_ids=())
    analysis = replace(configuration.analysis, context=replace(context, conditions=(*context.conditions, third, unused)))
    ui = replace(configuration.ui, preview_time_modes={"测试队列A": "window", "测试队列B": "cumulative"})
    projection = build_ui_projection(ui, analysis)

    assert projection.stations[1][1][-2:] == ("声压级（工况3）", "频谱（工况3）")
    assert tuple(binding.condition_id for binding in projection.bindings[1][1]) == ("c1", "c2")
    assert projection.preview_time_modes == {"c1": "window", "c2": "cumulative", "c3": "window", "c4": "window"}

    def assert_display_only(value):
        if is_dataclass(value):
            assert type(value).__module__.startswith("ui.sequence.")
            for field in fields(value):
                assert_display_only(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                assert isinstance(key, str)
                assert_display_only(item)
        elif isinstance(value, tuple):
            for item in value:
                assert_display_only(item)
        else:
            assert value is None or isinstance(value, str)

    assert_display_only(projection)
    with pytest.raises(FrozenInstanceError):
        projection.cells[0][0].unit = "changed"
    with pytest.raises(FrozenInstanceError):
        projection.bindings[0][0][0].condition_id = "changed"
