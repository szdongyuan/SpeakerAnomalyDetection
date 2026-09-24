from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ui.sequence.motor_result_panel import MotorResultPanel
from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin


def condition(key, selections):
    analysis = {"display_sequence": list(selections)}
    for name, (kind, channels) in selections.items():
        analysis[name] = {"type": kind, "analysis_channels": channels}
    return {"key": key, "condition_name": key, "analysis_list": analysis}


@pytest.fixture
def panel_factory(ui_qapp):
    panels = []

    def create(*conditions):
        panel = MotorResultPanel(condition_configs=list(conditions), queue_catalog={})
        panel.set_channels([0, 1, 2, 3, 4])
        panels.append(panel)
        return panel

    yield create
    for panel in panels:
        panel.close()
        panel.deleteLater()
    ui_qapp.processEvents()


@pytest.mark.parametrize("first,second,expected", [
    ([0, 1], [0, 1], [0, 1]),
    ([0, 1], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
    ([0], [4], [0, 4]),
    ([0, 2], [2, 4], [0, 2, 4]),
])
def test_each_condition_counts_union_of_selected_analysis_channels(panel_factory, first, second, expected):
    panel = panel_factory(condition("gear", {"a": ("SPL", first), "b": ("FBA", second)}))
    row = panel.rows["gear"]
    assert row["labels"]["progress"].text() == f"通道判定：0/{len(expected)}"
    assert len(panel.channel_detail_labels) == len(expected)
    # Hardware inputs remain available to waveform and progress persistence.
    assert panel.channel_indices == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("failure", ["分析失败", "结果不完整"])
@pytest.mark.parametrize("next_result", ["待检测", "准备采集", "分析中", "OK", "NG"])
def test_failure_retention_allows_explicit_new_test_or_result(panel_factory, failure, next_result):
    panel = panel_factory(condition("gear", {"SPL": ("SPL", [0, 1])}))
    panel.set_condition_result("gear", failure)
    panel.set_condition_result("gear", "待判定")
    assert panel.rows["gear"]["result"] == failure
    panel.set_condition_result("gear", "未标记")
    assert panel.rows["gear"]["result"] == failure
    panel.set_condition_result("gear", next_result)
    assert panel.rows["gear"]["result"] == next_result


def test_unselected_cells_are_dashes_and_do_not_block_channel_judgements(panel_factory):
    panel = panel_factory(condition("gear", {
        "a": ("SPL", [0, 1]), "b": ("FBA", [0, 1, 2, 3, 4]),
    }))
    assert [labels["SPL"].text() for labels in panel.channel_detail_labels] == [
        "待检测", "待检测", "—", "—", "—",
    ]
    results = [{"raw_channel": ch, "FBA": "OK", "result": "OK"} for ch in range(5)]
    for result in results[:2]:
        result["SPL"] = "OK"
    panel.set_condition_channel_results("gear", results)
    assert panel.rows["gear"]["labels"]["progress"].text() == "通道判定：5/5"
    assert panel.channel_detail_labels[2]["SPL"].text() == "—"
    assert panel.channel_detail_labels[2]["FBA"].text() == "OK"


def test_same_type_items_merge_channel_membership_without_double_counting(panel_factory):
    panel = panel_factory(condition("gear", {
        "pressure1": ("SPL", [0, 2]), "pressure2": ("SPL", [2, 4]),
    }))
    assert panel.rows["gear"]["channel_count"] == 3
    assert panel.channel_analysis_columns == [{"key": "SPL", "header": "SPL判定"}]
    assert [labels["SPL"].text() for labels in panel.channel_detail_labels] == ["待检测"] * 3


def test_condition_switches_and_out_of_order_results_keep_independent_channel_counts(panel_factory):
    panel = panel_factory(
        condition("two", {"a": ("SPL", [0, 4])}),
        condition("five", {"b": ("FBA", [0, 1, 2, 3, 4])}),
    )
    panel.select_condition("five", show_detail=True)
    panel.set_condition_channel_results("two", [
        {"raw_channel": 0, "SPL": "OK", "result": "OK"},
        {"raw_channel": 4, "SPL": "NG", "result": "NG"},
        {"raw_channel": 1, "SPL": "OK", "result": "OK"},
    ])
    assert len(panel.channel_detail_labels) == 5
    assert panel.rows["five"]["labels"]["progress"].text() == "通道判定：0/5"
    assert panel.rows["two"]["labels"]["progress"].text() == "通道判定：2/2"
    panel.select_condition("two", show_detail=True)
    assert [labels["result"].text() for labels in panel.channel_detail_labels] == ["OK", "NG"]


def test_configuration_refresh_changes_condition_channels_before_next_recording(panel_factory, monkeypatch):
    panel = panel_factory(condition("gear", {"a": ("SPL", [0, 1, 2, 3, 4])}))
    monkeypatch.setattr(panel, "_load_queue_catalog_safely", lambda: {})
    assert panel.refresh_condition_configs([condition("gear", {"a": ("SPL", [1, 3])})])
    assert panel.rows["gear"]["labels"]["progress"].text() == "通道判定：0/2"
    assert len(panel.channel_detail_labels) == 2
    panel.set_channels([0, 1, 2, 3, 4, 5])
    assert panel.rows["gear"]["channel_count"] == 2


def test_completed_result_does_not_replace_global_hardware_channels():
    left_panel = SimpleNamespace(
        set_channels=Mock(), set_condition_channel_results=Mock(), set_condition_result=Mock()
    )
    results = [{"raw_channel": 0, "result": "OK"}, {"raw_channel": 4, "result": "OK"}]
    host = SimpleNamespace(left_panel=left_panel, _build_process_channel_results=lambda _: results)
    result = SimpleNamespace(condition_key="two", final_judgement="OK")

    SequenceWidgetAnalysisProcessOpsMixin._sync_process_result_to_condition_panel(host, result, "OK", "ok")

    left_panel.set_channels.assert_not_called()
    left_panel.set_condition_channel_results.assert_called_once_with("two", results)


def test_same_table_shape_still_updates_cell_membership_when_switching_conditions(panel_factory):
    panel = panel_factory(
        condition("first", {"a": ("SPL", [0]), "b": ("FBA", [0, 1])}),
        condition("second", {"a": ("SPL", [0, 1]), "b": ("FBA", [1])}),
    )
    assert panel.channel_detail_labels[1]["SPL"].text() == "—"
    panel.select_condition("second", show_detail=True)
    assert panel.channel_detail_labels[1]["SPL"].text() == "待检测"
    assert panel.channel_detail_labels[0]["FBA"].text() == "—"
    panel.select_condition("first", show_detail=True)
    assert panel.channel_detail_labels[1]["SPL"].text() == "—"
    assert panel.channel_detail_labels[0]["FBA"].text() == "待检测"


def test_legacy_single_channel_config_uses_physical_channel_number(panel_factory):
    config = condition("gear", {"a": ("SPL", [0])})
    config["analysis_list"]["a"] = {"type": "SPL", "analysis_channel": 3}
    panel = panel_factory(config)
    assert panel.rows["gear"]["channel_count"] == 1
    panel.set_condition_channel_results("gear", [{"raw_channel": 3, "SPL": "OK", "result": "OK"}])
    assert panel.channel_grid.itemAtPosition(1, 0).widget().text() == "CH4"
    assert panel.channel_detail_labels[0]["SPL"].text() == "OK"


def test_empty_analysis_queue_does_not_claim_five_pending_channels(panel_factory):
    panel = panel_factory(condition("gear", {}))
    assert panel.rows["gear"]["labels"]["progress"].text() == "通道判定：0/0"
    assert panel.channel_detail_labels == []


def test_channel_sync_preserves_late_results_and_uses_new_channels_after_reset(panel_factory):
    old = condition("gear", {"a": ("SPL", [0, 4])})
    new = condition("gear", {"a": ("SPL", [1])})
    panel = panel_factory(old)
    panel.set_condition_result("gear", "分析中", tone="running")
    assert panel.refresh_condition_configs([new], queue_catalog={}, preserve_results=True)
    panel.set_condition_channel_results("gear", [
        {"raw_channel": ch, "SPL": "OK", "result": "OK"} for ch in [0, 4]])
    panel.set_condition_result("gear", "OK")
    assert panel.rows["gear"]["labels"]["progress"].text() == "通道判定：2/2"
    assert panel.rows["gear"]["analysis_channels"] == [0, 4]
    panel.reset()
    assert panel.rows["gear"]["labels"]["progress"].text() == "通道判定：0/1"
    assert panel.rows["gear"]["analysis_channels"] == [1]


def test_rerecording_uses_new_channels_without_resetting_other_conditions(panel_factory):
    first = condition("first", {"a": ("SPL", [0, 4])})
    second = condition("second", {"a": ("SPL", [0, 4])})
    panel = panel_factory(first, second)
    panel.set_condition_result("first", "NG")
    panel.set_condition_result("second", "OK")
    panel.refresh_condition_configs([
        condition("first", {"a": ("SPL", [1])}),
        condition("second", {"a": ("SPL", [1])}),
    ], queue_catalog={}, preserve_results=True)
    panel.set_condition_result("first", "准备采集", tone="running")
    assert panel.rows["first"]["analysis_channels"] == [1]
    assert panel.rows["second"]["analysis_channels"] == [0, 4]
    assert panel.rows["second"]["result"] == "OK"
