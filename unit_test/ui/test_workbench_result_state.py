"""Single-channel display results and recording-day paired terminal records."""

from datetime import date, datetime
from itertools import product

import pytest

from base.analysis.contracts import AnalysisInstance
from base.analysis.results import calculation_payload
from base.motor_daily_records import DailyRecordStore
from ui.sequence.workbench_result_state import WorkbenchResultState
from ui.sequence.workbench_ui_projection import _plot_unit


def instance(key="SPL", kind="SPL", judgement="OK", *, metrics=None,
             execution_status="分析完成"):
    return {
        "config_key": key, "analysis_type": kind, "channel_index": 0,
        "execution_status": execution_status, "judgement_status": "judged",
        "judgement": judgement, "metrics": {} if metrics is None else metrics,
        "display_payload": None, "violations": None, "artifacts": [], "error": None,
    }


def payload(condition="forward", items=(), station="station-a", *, failed=False):
    return {
        "task_id": f"task-{station}-{condition}", "station_id": station,
        "condition_id": condition,
        "execution_status": "分析失败" if failed else "分析完成",
        "instance_results": list(items),
        "error": {"stage": "wav_read", "type": "OSError", "message": "missing"}
        if failed else None,
    }


def open_round(state, round_id="round-1", *, recorded_at=None, stations=None,
               product_code="007", source_identity="product.json"):
    state.open_round(
        round_id, analysis_version="v1", source_identity=source_identity,
        recorded_at=recorded_at or datetime(2026, 11, 11, 23, 59, 59),
        condition_ids=("forward", "reverse"),
        stations=stations if stations is not None else (("station-a", 1, ("SPL", "FFT")),),
        product_code=product_code,
    )


@pytest.mark.parametrize("values,expected", [
    ([], "--"), ([None, 0, False, {}, [], "", "unknown", "--"], "--"),
    ([" ok ", None], "OK"), (["\tng\n", "OK"], "NG"),
] + [(list(values), "NG" if "NG" in values else "OK" if "OK" in values else "--")
     for values in product(("OK", "NG", "--"), repeat=3)])
def test_direction_status_aggregates_all_returned_judgements(values, expected):
    state = WorkbenchResultState()
    open_round(state)
    items = [instance(f"key-{i}", judgement=value) for i, value in enumerate(values)]
    result = state.receive("round-1", "station-a", "forward", payload(items=items))
    assert (result.forward, result.reverse, result.result) == (expected, "--", "--")
    assert result.complete is False


@pytest.mark.parametrize("kind,metrics,execution_status,expected", [
    ("SPL", {"overall_spl": 62.3, "unit": "dB(A)"}, "分析完成", "62.30 dB(A)"),
    ("FBA", {"overall_weighted_db": 63.125, "unit": "dB(C)"}, "分析完成", "63.12 dB(C)"),
    ("SPL", {"overall_spl": 0, "unit": "dB(Z)"}, "分析完成", "0.00 dB(Z)"),
    ("SPL", {"overall_spl": 63.13, "unit": "dBA"}, "分析完成", "63.13 dBA"),
    ("FBA", {"overall_weighted_db": 63.13, "unit": "dB"}, "分析完成", "63.13 dB"),
    ("SPL", {"overall_spl": -2}, "分析完成", "-2.00"),
    ("FBA", {"overall_weighted_db": 1, "unit": None}, "分析完成", "1.00"),
    ("FFT", {"peak_frequency_hz": 1000, "peak_db": 90, "overall_spl": 90}, "分析完成", "分析完成"),
    ("Spec", {"sample_rate": 44100}, "分析完成", "分析完成"),
    ("SPL", {}, "分析完成", "分析完成"),
    ("SPL", {"overall_spl": 60}, "分析失败", "--"),
] + [("SPL", {"overall_spl": value}, "分析完成", "分析完成")
     for value in (None, float("nan"), float("inf"), -float("inf"), True, "12")])
def test_metric_projection_uses_only_completed_finite_overall_value(
        kind, metrics, execution_status, expected):
    state = WorkbenchResultState()
    open_round(state)
    item = instance(kind=kind, judgement=None, metrics=metrics,
                    execution_status=execution_status)
    result = state.receive("round-1", "station-a", "forward", payload(items=[item]))
    assert result.rows == [["SPL", expected, "--"], ["FFT", "--", "--"]]
    assert result.result == "--"  # Successful execution never implies OK.


@pytest.mark.parametrize("kind,key", [("SPL", "overall_spl"), ("FBA", "overall_weighted_db")])
@pytest.mark.parametrize("weighting,expected_unit", [
    ("A", "dB(A)"), ("b", "dB(B)"), ("C", "dB(C)"), ("D", "dB(D)"),
    ("Z", "dB(Z)"), (None, "dB(Z)"), ("None", "dB(Z)"), ("Z（None）", "dB(Z)"),
])
@pytest.mark.parametrize("old_unit", ["dBA", "dB"])
def test_actual_weighting_overrides_legacy_unit_and_matches_plot(kind, key, weighting, expected_unit, old_unit):
    state = WorkbenchResultState()
    open_round(state)
    item = instance(kind=kind, metrics={key: 63.13, "weighting": weighting, "unit": old_unit})
    result = state.receive("round-1", "station-a", "forward", payload(items=[item]))
    assert result.rows[0][1] == f"63.13 {expected_unit}"
    assert expected_unit == _plot_unit({"weighting": weighting})


@pytest.mark.parametrize("kind,key,weighting", [
    ("SPL", "overall_spl", "Z"), ("SPL", "overall_spl", "B"),
    ("FBA", "overall_weighted_db", "Z"), ("FBA", "overall_weighted_db", "D"),
])
def test_real_calculation_payload_displays_actual_weighting(kind, key, weighting):
    analysis_instance = AnalysisInstance("SPL", kind, 0, {"weighting": weighting})
    calculation = {"metrics": {key: 63.13}, "curve": {"x": [0], "y": [63.13]},
                   "plot": {"labels": ["1000"]}}
    metrics, display_payload = calculation_payload(analysis_instance, calculation, 0)
    item = instance(kind=kind, metrics=metrics)
    item["display_payload"] = display_payload
    state = WorkbenchResultState()
    open_round(state)
    result = state.receive("round-1", "station-a", "forward", payload(items=[item]))
    assert result.rows[0][1] == f"63.13 dB({weighting})"
    assert result.rows[0][1] == f"63.13 {_plot_unit(analysis_instance.parameters)}"


def test_values_and_judgements_survive_other_instance_and_task_failures():
    state = WorkbenchResultState()
    open_round(state)
    items = [instance(metrics={"overall_spl": 50}, judgement="NG"),
             instance("FFT", "FFT", None, execution_status="分析失败")]
    result = state.receive("round-1", "station-a", "forward", payload(items=items, failed=True))
    assert result.forward == "NG"
    assert result.rows == [["SPL", "50.00", "--"], ["FFT", "--", "--"]]


def test_config_keys_keep_same_type_items_separate_in_frozen_order():
    state = WorkbenchResultState()
    keys = ["SPL-A", "SPL-B", "FFT"]
    open_round(state, stations=[["station-a", 1, keys]])
    keys[:] = ["changed"]
    items = [instance("SPL-B", metrics={"overall_spl": 20}),
             instance("SPL-A", metrics={"overall_spl": 10})]
    result = state.receive("round-1", "station-a", "reverse", payload("reverse", items))
    assert result.rows == [["SPL-A", "--", "10.00"], ["SPL-B", "--", "20.00"],
                           ["FFT", "--", "--"]]


@pytest.mark.parametrize("forward,reverse", product(("OK", "NG", "--"), repeat=2))
@pytest.mark.parametrize("reverse_first", [False, True])
def test_pair_counts_only_after_both_terminals_with_recording_timestamp(
        forward, reverse, reverse_first):
    state = WorkbenchResultState()
    recorded_at = datetime(2026, 11, 11, 23, 59, 59)
    open_round(state, recorded_at=recorded_at)
    directions = [("forward", forward), ("reverse", reverse)]
    if reverse_first:
        directions.reverse()
    first_direction, first_status = directions[0]
    first = state.receive("round-1", "station-a", first_direction,
                          payload(first_direction, [instance(judgement=first_status)]))
    assert first.complete is False
    assert first.result == "--"
    assert getattr(first, first_direction) == first_status
    assert state.records_for(recorded_at.date()) == []
    last_direction, last_status = directions[1]
    last = state.receive("round-1", "station-a", last_direction,
                         payload(last_direction, [instance(judgement=last_status)]))
    assert last.complete is True
    with pytest.raises(AttributeError):
        last.complete = False
    expected = "NG" if "NG" in (forward, reverse) else "OK" if "OK" in (forward, reverse) else "--"
    assert (last.forward, last.reverse, last.result) == (forward, reverse, expected)
    assert state.records_for(recorded_at.date()) == [{
        "completed_at": recorded_at, "station_id": 1, "product_code": "007",
        "forward": forward, "reverse": reverse,
        "result": "未标记" if expected == "--" else expected,
    }]
    assert state.records_for(date(2026, 11, 12)) == []


def test_failed_terminal_still_completes_pair_and_duplicate_never_overwrites():
    state = WorkbenchResultState()
    open_round(state)
    first = state.receive("round-1", "station-a", "forward", payload(failed=True))
    assert first.complete is False
    assert state.receive("round-1", "station-a", "forward", payload(items=[instance()])) is None
    assert state.records_for(date(2026, 11, 11)) == []
    last = state.receive("round-1", "station-a", "reverse", payload("reverse", failed=True))
    assert last.complete is True
    assert last.forward == last.reverse == last.result == "--"
    assert state.receive("round-1", "station-a", "reverse", payload("reverse", [instance()])) is None
    assert len(state.records_for(date(2026, 11, 11))) == 1
    assert state.records_for(date(2026, 11, 11))[0]["result"] == "未标记"


def test_stations_rounds_and_recording_days_complete_independently():
    state = WorkbenchResultState()
    stations = (("station-a", 2, ("SPL",)), ("station-b", 4, ()))
    open_round(state, stations=stations)
    state.receive("round-1", "station-a", "forward", payload(items=[instance()]))
    open_round(state, "round-2", stations=stations, recorded_at=datetime(2026, 11, 12),
               product_code="new", source_identity="new.json")
    current = state.receive("round-2", "station-a", "forward", payload(items=[instance()]))
    assert current.reverse == "--" and current.rows == [["SPL", "分析完成", "--"]]
    old = state.receive("round-1", "station-a", "reverse", payload("reverse", [instance(judgement="NG")]))
    assert old.result == "NG"
    for direction in ("reverse", "forward"):
        empty = state.receive("round-1", "station-b", direction, payload(direction, station="station-b"))
        assert empty.rows == []
    assert [record["station_id"] for record in state.records_for(date(2026, 11, 11))] == [2, 4]
    assert state.records_for(date(2026, 11, 12)) == []
    current = state.receive("round-2", "station-a", "reverse", payload("reverse", [instance()]))
    assert current.result == "OK"
    assert state.records_for(date(2026, 11, 12))[0]["product_code"] == "new"


def test_reopening_round_and_mutating_snapshots_cannot_reset_or_change_state():
    state = WorkbenchResultState()
    open_round(state)
    original = payload(items=[instance(metrics={"overall_spl": 12})])
    first = state.receive("round-1", "station-a", "forward", original)
    first.rows[0][1] = "caller mutation"
    first.rows.clear()
    original["instance_results"][0]["metrics"]["overall_spl"] = 999
    open_round(state, recorded_at=datetime(2026, 11, 12), product_code="changed")
    last = state.receive("round-1", "station-a", "reverse", payload("reverse"))
    assert last.forward == "OK" and last.rows[0] == ["SPL", "12.00", "--"]
    records = state.records_for(date(2026, 11, 11))
    records[0]["product_code"] = "changed"
    records.append({})
    assert state.records_for(date(2026, 11, 11))[0]["product_code"] == "007"
    assert len(state.records_for(date(2026, 11, 11))) == 1
    assert WorkbenchResultState().records_for(date(2026, 11, 11)) == []


@pytest.mark.parametrize("round_id,station,condition,changed_field,changed_value,diagnostic", [
    ("unknown-round", "station-a", "forward", None, None, "unknown-round"),
    ("round-1", "unknown-station", "forward", None, None, "unknown-station"),
    ("round-1", "station-a", "unknown-condition", None, None, "unknown-condition"),
    ("round-1", "station-a", "forward", "station_id", "wrong-station", "wrong-station"),
    ("round-1", "station-a", "forward", "condition_id", "wrong-condition", "wrong-condition"),
])
def test_invalid_identity_reports_context_without_consuming_terminal(
        round_id, station, condition, changed_field, changed_value, diagnostic):
    state = WorkbenchResultState()
    open_round(state)
    incoming = payload()
    if changed_field:
        incoming[changed_field] = changed_value
    with pytest.raises(ValueError, match=diagnostic) as error:
        state.receive(round_id, station, condition, incoming)
    assert "round" in str(error.value)
    if changed_field:
        assert incoming["task_id"] in str(error.value)
    assert state.records_for(date(2026, 11, 11)) == []
    assert state.receive("round-1", "station-a", "forward", payload()) is not None


def test_daily_record_store_accepts_completed_day_snapshot_and_counts_categories():
    state = WorkbenchResultState()
    for round_id, status in (("round-ok", "OK"), ("round-ng", "NG"), ("round-unmarked", "--")):
        open_round(state, round_id)
        for direction in ("forward", "reverse"):
            state.receive(round_id, "station-a", direction, payload(direction, [instance(judgement=status)]))
    store = DailyRecordStore()
    day = date(2026, 11, 11)
    store.replace(state.records_for(day), day)
    assert (store.summary.total, store.summary.ok, store.summary.ng,
            store.summary.unmarked, store.summary.pass_rate) == (3, 1, 1, 1, "33.3%")
    next_day = date(2026, 11, 12)
    store.replace(state.records_for(next_day), next_day)
    assert (store.summary.total, store.summary.pass_rate) == (0, "--")
