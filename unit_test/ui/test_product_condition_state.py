"""Business regressions for condition state without a task widget."""

from copy import deepcopy

import pytest
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QWidget

from ui.sequence.product_condition_state import ProductConditionState


def condition(key, selections=None):
    config = {"key": key, "name": key}
    if selections is not None:
        config["analysis_list"] = {"display_sequence": list(selections)}
        for name, (kind, channels) in selections.items():
            config["analysis_list"][name] = {"type": kind, "analysis_channels": channels}
    return config


def make_state(*configs):
    return ProductConditionState(condition_configs=list(configs), queue_catalog={})


def test_state_is_independent_qobject_with_business_rows_only():
    state = make_state(condition("a"), condition("b"))
    assert isinstance(state, QObject)
    assert not isinstance(state, QWidget)
    assert not state.findChildren(QWidget)
    assert all("button" not in row and "labels" not in row for row in state.rows.values())
    state.set_condition_result("a", "OK")
    state.set_condition_result("b", "NG")
    assert state.get_automatic_round_result() == ("NG", "ng", True)
    assert (state.final_result, state.final_tone) == ("NG", "ng")
    other = make_state(condition("a"))
    assert other.rows["a"]["result"] == "待检测"


@pytest.mark.parametrize("results, expected", [
    ([], ("待判定", "pending", False)),
    (["OK", "待检测"], ("待判定", "pending", False)),
    (["NG", "待检测"], ("NG", "ng", True)),
    (["NG", "结果不完整"], ("结果不完整", "ng", True)),
    (["分析失败", "OK"], ("待判定", "pending", False)),
    (["OK", "OK"], ("OK", "ok", True)),
    (["未判定", "OK"], ("未产生判定", "pending", True)),
])
def test_automatic_round_result_preserves_business_priority(results, expected):
    state = make_state(*(condition(str(index)) for index in range(len(results))))
    for index, result in enumerate(results):
        state.set_condition_result(index, result)
    assert state.get_automatic_round_result() == expected


@pytest.mark.parametrize("failure", ["分析失败", "结果不完整"])
@pytest.mark.parametrize("placeholder", ["待判定", "未标记"])
def test_pending_recording_placeholder_does_not_overwrite_failure(failure, placeholder):
    state = make_state(condition("gear"))
    state.set_condition_result("gear", failure)
    state.set_condition_result("gear", placeholder)
    assert state.rows["gear"]["result"] == failure
    assert state.rows["gear"]["tone"] == "ng"
    assert state.rows["gear"]["analysis_completed"] is (failure == "结果不完整")
    state.set_condition_result("gear", "准备采集")
    assert state.rows["gear"]["result"] == "准备采集"
    assert state.rows["gear"]["analysis_completed"] is False


@pytest.mark.parametrize("results, expected", [
    (["OK", "OK"], ("OK", "ok")),
    (["NG", "结果不完整"], ("结果不完整", "ng")),
    (["NG", "待判定"], ("NG", "ng")),
    (["OK", "待判定"], ("未判定", "pending")),
    ([], ("未判定", "pending")),
])
def test_completed_analysis_placeholders_use_channel_verdicts(results, expected):
    state = make_state(condition("gear"))
    state.set_channels([0, 1])
    state.set_condition_channel_results("gear", [{"result": result} for result in results])
    state.set_condition_result("gear", "未判定")
    state.set_condition_result("gear", "未标记")
    assert (state.rows["gear"]["result"], state.rows["gear"]["tone"]) == expected


def test_analysis_layout_unions_channels_and_filters_duplicate_results():
    state = make_state(condition("gear", {
        "pressure": ("SPL", [0, 2]), "pressure2": ("SPL", [2, 4]),
        "bands": ("FBA", [4]),
    }))
    state.set_channels([0, 1, 2, 3, 4, 5])
    row = state.rows["gear"]
    assert row["analysis_channels"] == [0, 2, 4]
    assert row["analysis_column_channels"] == {"SPL": [0, 2, 4], "FBA": [4]}
    state.set_condition_channel_results("gear", [
        {"raw_channel": 4, "result": "OK"}, {"raw_channel": 4, "result": "NG"},
        {"raw_channel": 1, "result": "OK"}, {"raw_channel": 2, "result": "待判定"},
        {"raw_channel": 0, "result": " ok "},
    ])
    assert row["channel_count"] == 3
    assert row["completed_channels"] == 2
    assert {item["raw_channel"]: item["result"] for item in row["channel_results"]} == {
        4: "NG", 2: "待判定", 0: " ok ",
    }
    assert state.channel_indices == [0, 1, 2, 3, 4, 5]


def test_hardware_fallback_and_valid_empty_analysis_are_distinct():
    state = make_state(condition("fallback"), condition("empty", {}))
    state.set_channels([2, 7])
    assert state.rows["fallback"]["channel_count"] == 2
    assert state.rows["empty"]["channel_count"] == 0
    state.set_condition_channel_results("fallback", [{"result": "OK"}, {"result": "NG"}])
    assert [item["raw_channel"] for item in state.rows["fallback"]["channel_results"]] == [2, 7]
    state.set_channels([2])
    assert state.rows["fallback"]["completed_channels"] == 1
    assert state.rows["empty"]["channel_results"] == []


@pytest.mark.parametrize("historical, expected", [(None, [0, 4]), ([0, 2, 4], [0, 2, 4])])
def test_restoration_uses_historical_channels_instead_of_current_config(historical, expected):
    state = make_state(condition("gear", {"pressure": ("SPL", [1])}))
    state.set_condition_channel_results("gear", [
        {"raw_channel": 4, "SPL": "NG", "result": "NG"},
        {"raw_channel": 0, "SPL": "OK", "result": "OK"},
    ], restore_channels=True, restored_analysis_channels=historical,
        restored_analysis_column_channels={"SPL": [0, 4]})
    row = state.rows["gear"]
    assert row["analysis_channels"] == expected
    assert row["analysis_column_channels"] == {"SPL": [0, 4]}
    assert row["channel_count"] == len(expected)
    assert row["completed_channels"] == 2
    assert state.channel_indices == [0]


def test_restoring_empty_historical_layout_and_missing_legacy_mapping():
    state = make_state(condition("gear", {"pressure": ("SPL", [1])}))
    state.set_condition_channel_results("gear", [], restore_channels=True, restored_analysis_channels=[])
    assert state.rows["gear"]["analysis_channels"] == []
    assert state.rows["gear"]["analysis_column_channels"] == {}
    assert state.rows["gear"]["channel_count"] == 0


def test_refresh_keeps_in_progress_layout_until_rerecord_or_batch_reset():
    old = [condition(key, {"pressure": ("SPL", [0, 4])}) for key in ("a", "b")]
    updated = [condition(key, {"pressure": ("SPL", [1])}) for key in ("a", "b")]
    state = make_state(*old)
    state.set_condition_result("a", "分析中")
    state.set_condition_result("b", "OK")
    assert state.refresh_condition_configs(updated, queue_catalog={}, preserve_results=True)
    state.set_condition_channel_results("a", [{"raw_channel": 4, "result": "NG"}])
    assert state.rows["a"]["completed_channels"] == 1
    assert state.rows["a"]["analysis_channels"] == [0, 4]
    state.set_condition_result("a", "准备采集")
    assert state.rows["a"]["analysis_channels"] == [1]
    assert state.rows["b"]["analysis_channels"] == [0, 4]
    assert state.rows["b"]["result"] == "OK"
    state.reset()
    assert all(row["analysis_channels"] == [1] for row in state.rows.values())
    assert all(row["result"] == "待检测" and not row["analysis_completed"] for row in state.rows.values())
    assert all(row["channel_results"] == [] and row["completed_channels"] == 0 for row in state.rows.values())
    assert (state.final_result, state.final_tone) == ("待判定", "pending")
    assert state.selected_key == "a"


def test_refresh_updates_unstarted_layout_and_rejects_changed_keys_atomically():
    state = make_state(condition("gear", {"p": ("SPL", [0, 4])}))
    state.set_condition_channel_results("gear", [{"raw_channel": 4, "result": "OK"}])
    assert state.refresh_condition_configs([condition("gear", {"p": ("SPL", [1])})], queue_catalog={})
    assert state.rows["gear"]["channel_count"] == 1
    assert state.rows["gear"]["channel_results"] == []
    before = deepcopy((state.conditions, state.rows))
    assert not state.refresh_condition_configs([condition("other")], queue_catalog={})
    assert (state.conditions, state.rows) == before


def test_runtime_details_merge_normalize_and_clear_without_touching_results():
    state = make_state(condition("gear"))
    assert state.set_condition_analysis_details("gear", {"spl": " 42 dB ", "FFT": 10, "unknown": "bad"})
    state.set_condition_analysis_details(0, [
        {"label": "fba", "value": "NG"}, {"label": "SPL", "value": None},
    ])
    assert state.rows["gear"]["runtime_details"] == {"FFT": "10", "FBA": "NG"}
    state.set_condition_analysis_details("gear", {"FFT": None})
    assert state.rows["gear"]["runtime_details"] == {"FBA": "NG"}
    state.set_condition_result("gear", "待检测")
    assert state.rows["gear"]["runtime_details"] == {}
    state.set_condition_analysis_details("gear", {"SPL": "OK"})
    state.reset()
    assert state.rows["gear"]["runtime_details"] == {}
    assert state.set_condition_analysis_details("unknown", {"SPL": "NG"}) is False


@pytest.mark.parametrize("raw, expected, tone", [
    ("", "等待开始", "pending"), ("采集中", "", "pending"),
    ("A / 6000 数据保存中", "数据保存中", "running"),
    ("A / 6000 分析中", "分析中", "running"),
    ("分析排队", "分析中", "running"),
    ("等待7000档位状态码", "等待下一档位", "pending"),
    ("A / 6000 分析失败", "测试异常", "ng"),
    ("本轮采集完成", "本轮完成", "ok"),
    ("未知阶段", "测试异常", "ng"),
])
def test_stage_transitions_use_plain_business_fields(raw, expected, tone):
    state = make_state()
    state.set_current_stage(raw, tone="ok")
    assert (state.stage_text, state.stage_tooltip, state.stage_tone) == (expected, expected, tone)
    state.stage_tooltip = "历史进度恢复提示"
    state.clear_current_stage()
    assert (state.stage_text, state.stage_tooltip, state.stage_tone) == ("", "", "pending")


def test_active_selection_notifications_follow_recording_transitions():
    state = make_state(condition("a"), condition("b"))
    selected = []
    state.condition_selected.connect(selected.append)
    state.select_condition("b", user_view=False)
    assert state.selected_key == "b"
    assert state.viewed_key == ""
    assert selected == ["b"]
    state.set_condition_result("a", "采集中")
    state.set_condition_result("a", "采集中")
    state.set_condition_result("b", "OK")
    assert state.selected_key == "a"
    assert selected == ["b", "a"]
    state.select_condition("missing")
    assert selected == ["b", "a"]
    state.reset()
    assert selected == ["b", "a", "a"]


def test_configuration_normalization_queue_catalog_and_legacy_channel():
    state = ProductConditionState(condition_configs=[
        None, {"name": " "},
        {"key": "same", "group_name": " A ", "condition_name": " 正转 ", "test_queue": "queue"},
        {"key": "same", "name": "反转", "analysis_list": {"display_sequence": ["p"], "p": {
            "type": "SPL", "analysis_channel": 3,
        }}},
    ], queue_catalog={"queue": {"analysis_list": {"display_sequence": ["loud"], "loud": {
        "type": "LOUD", "analysis_channels": [0, 2],
    }}}})
    assert state.condition_names == ["A / 正转", "反转"]
    assert list(state.rows) == ["same", "same#4"]
    assert state.rows["same"]["analysis_channels"] == [0, 2]
    assert state.rows["same#4"]["analysis_channels"] == [3]
    assert state.set_condition_result("a / 正转", "NG")
    assert state.set_reverse_result("OK")
    assert not state.set_condition_result("missing", "NG")
    state.set_condition_configs([condition("new")], queue_catalog={})
    assert list(state.rows) == ["new"]
    assert state.selected_key == "new"


@pytest.mark.parametrize("value, expected", [(0, 1), ("3", 3), (None, 1), ("bad", 1)])
def test_round_coercion_preserves_legacy_behavior(value, expected):
    state = make_state(condition("a"))
    state.set_current_round(value)
    assert state.current_round == expected
    state.reset()
    assert state.current_round == expected


@pytest.mark.parametrize("placeholder", ["检测中", "未标记", "完成", "待判定"])
def test_final_placeholder_retains_resolved_automatic_verdict(placeholder):
    state = make_state(condition("a"))
    state.set_final_result(placeholder)
    assert state.final_result == "待判定"
    state.set_forward_result("OK")
    state.set_final_result(placeholder)
    assert (state.final_result, state.final_tone) == ("OK", "ok")


@pytest.mark.parametrize("previous", ["分析失败", "结果不完整", "OK", "NG"])
@pytest.mark.parametrize("placeholder", ["待判定", "未标记"])
def test_migrated_visual_row_placeholder_scenarios_keep_business_outcomes(previous, placeholder):
    state = make_state(condition("a", {"SPL": ("SPL", [0])}))
    if previous in ("OK", "NG"):
        state.set_condition_channel_results("a", [{"raw_channel": 0, "result": previous}])
    state.set_condition_result("a", previous)
    state.set_condition_result("a", placeholder)
    assert state.rows["a"]["result"] == previous
    assert state.rows["a"]["tone"] == ("ok" if previous == "OK" else "ng")


def test_migrated_reset_rebuild_and_empty_configuration():
    configs = [condition(key) for key in ("a", "b", "c")]
    state = make_state(*configs)
    state.select_condition("a")
    state.set_condition_result("b", "NG")
    rows = state.rows
    assert state.refresh_condition_configs(configs, queue_catalog={}, preserve_results=True)
    assert state.rows is rows
    assert state.rows["b"]["result"] == "NG"
    state.reset()
    assert state.viewed_key == ""
    assert all(row["result"] == "待检测" for row in state.rows.values())
    state.select_condition("b")
    state.set_condition_configs(configs, queue_catalog={})
    assert state.rows is not rows
    state.set_condition_configs([], queue_catalog={})
    state.reset()
    assert state.rows == {}
    assert state.selected_key == state.viewed_key == ""
    assert not state.set_condition_result("missing", "OK")


def test_migrated_200_condition_queue_cache_and_batch_reset(monkeypatch):
    configs = [{"key": str(i), "condition_name": str(i), "test_queue": "same"} for i in range(200)]
    catalog = {"same": {"analysis_list": {"display_sequence": ["SPL"], "SPL": {"type": "SPL", "analysis_channels": [0, 3, 8]}}}}
    from unittest.mock import patch
    with patch.object(ProductConditionState, "_build_condition_analysis_layout",
                      wraps=ProductConditionState._build_condition_analysis_layout) as layouts:
        state = ProductConditionState(condition_configs=configs, queue_catalog=catalog)
    layouts.assert_called_once()
    state.set_channels([0, 3, 8])
    state.set_condition_result("0", "NG")
    with patch.object(state, "set_condition_result", wraps=state.set_condition_result) as individual:
        state.reset()
    individual.assert_not_called()
    assert len(state.rows) == 200
    for row in state.rows.values():
        assert row["result"] == "待检测"
        assert row["channel_results"] == []
        assert row["runtime_details"] == {}
        assert not row["analysis_completed"]
        assert (row["completed_channels"], row["channel_count"]) == (0, 3)
    assert state.selected_key == "0"
    assert state.final_result == "待判定"


def test_recording_next_condition_preserves_previous_channel_results_and_details():
    state = make_state(condition("a", {"pressure": ("SPL", [0, 1])}),
                       condition("b", {"pressure": ("SPL", [0, 1])}))
    results = [{"raw_channel": 0, "SPL": "OK", "result": "OK"},
               {"raw_channel": 1, "SPL": "NG", "result": "NG"}]
    state.set_condition_channel_results("a", results)
    state.set_condition_result("a", "NG")
    state.set_condition_analysis_details("a", {"SPL": "previous"})
    before = deepcopy(state.rows["a"])
    state.select_condition("a")
    state.set_condition_result("b", "采集中", tone="running")
    assert state.selected_key == "b"
    assert state.viewed_key == ""
    assert state.rows["a"] == before
    assert state.rows["b"]["channel_results"] == []
    assert state.rows["b"]["completed_channels"] == 0
    state.select_condition("a")
    assert state.rows["a"]["channel_results"] == results
