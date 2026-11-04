"""Behavioral contract for caller-owned daily motor records."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from base.motor_daily_records import DailyRecordStore
from unit_test.daily_record_cases import make_daily_records


@pytest.fixture
def local_day(monkeypatch):
    class LocalDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 4)

    monkeypatch.setattr("base.motor_daily_records.date", LocalDate)
    return LocalDate.today()


def assert_invalid_record_is_atomic(field, value, error):
    store = DailyRecordStore()
    old_day = date(2026, 11, 4)
    store.replace(make_daily_records(3, old_day), old_day)
    before = (store.record_date, store.records, store.summary, store.station_ids)
    records = make_daily_records(2, date(2026, 11, 3))
    records[1][field] = value
    with pytest.raises(error, match=rf"records\[1\].*{field}"):
        store.replace(records, date(2026, 11, 3))
    assert (store.record_date, store.records, store.summary, store.station_ids) == before


def test_replace_sorts_newest_first_and_counts_final_results(local_day):
    day = local_day
    records = make_daily_records(3, day)
    records[0]["forward"] = "NG"
    records[0]["result"] = " ok "
    store = DailyRecordStore()
    assert store.record_date == day
    assert store.records == ()
    assert store.station_ids == ()
    assert store.summary.total == 0
    assert store.summary.pass_rate == "--"
    assert store.replace(records, day) is None
    assert store.record_date == day
    assert [r.completed_at for r in store.records] == sorted(
        (r["completed_at"] for r in records), reverse=True
    )
    assert (store.summary.total, store.summary.ok, store.summary.ng,
            store.summary.unmarked, store.summary.pass_rate) == (3, 1, 1, 1, "33.3%")


@pytest.mark.parametrize("records", [None, "records", {}, iter(())])
def test_rejects_non_list_or_tuple_batch(records):
    store = DailyRecordStore()
    with pytest.raises(TypeError, match="records"):
        store.replace(records, date(2026, 11, 4))


@pytest.mark.parametrize("day", ["2026-11-04", datetime(2026, 11, 4), 1])
def test_rejects_invalid_record_date_even_with_empty_batch(day):
    with pytest.raises(TypeError, match="record_date"):
        DailyRecordStore().replace([], day)


@pytest.mark.parametrize("item", [None, [], "record", 1])
def test_rejects_non_dictionary_item_with_index(item):
    with pytest.raises(TypeError, match=r"records\[0\]"):
        DailyRecordStore().replace([item], date(2026, 11, 4))


@pytest.mark.parametrize("field", [
    "completed_at", "station_id", "product_code", "forward", "reverse", "result",
])
def test_missing_field_reports_index_and_field_without_changing_snapshot(field):
    day = date(2026, 11, 4)
    store = DailyRecordStore()
    store.replace(make_daily_records(2, day), day)
    before = (store.record_date, store.records, store.summary, store.station_ids)
    records = make_daily_records(2, date(2026, 11, 3))
    del records[1][field]
    with pytest.raises(ValueError, match=rf"records\[1\].*{field}"):
        store.replace(records, date(2026, 11, 3))
    assert (store.record_date, store.records, store.summary, store.station_ids) == before


def test_tuple_batch_extra_fields_empty_product_and_default_date_are_accepted(local_day):
    records = make_daily_records(1, local_day)
    records[0].update(product_code="", extra={"ignored": True})
    store = DailyRecordStore()
    store.replace(tuple(records))
    assert store.record_date == local_day
    assert store.records[0].product_code == ""
    assert set(vars(store.records[0])) == set(records[0]) - {"extra"}


@pytest.mark.parametrize("field,value,error", [
    ("completed_at", date(2026, 11, 3), TypeError),
    ("completed_at", "12:00:00", TypeError),
    ("completed_at", None, TypeError),
    ("completed_at", datetime(2026, 11, 3, tzinfo=timezone.utc), ValueError),
    ("completed_at", datetime(2026, 11, 4), ValueError),
    ("station_id", True, TypeError),
    ("station_id", False, TypeError),
    ("station_id", 1.0, TypeError),
    ("station_id", "1", TypeError),
    ("station_id", None, TypeError),
    ("station_id", 0, ValueError),
    ("station_id", -1, ValueError),
    ("product_code", 123, TypeError),
    ("product_code", None, TypeError),
    ("product_code", b"code", TypeError),
])
def test_invalid_record_values_are_diagnostic_and_batch_atomic(field, value, error):
    assert_invalid_record_is_atomic(field, value, error)


def test_full_time_precision_and_sorted_distinct_station_ids():
    day = date(2026, 11, 4)
    records = make_daily_records(3, day)
    records[0]["station_id"] = 9
    records[1]["station_id"] = 9
    store = DailyRecordStore()
    store.replace(records, day)
    assert store.station_ids == (3, 9)
    assert [row.product_code for row in store.records] == ["CASE-003", "CASE-002", "CASE-001"]


@pytest.mark.parametrize("field", ["forward", "reverse", "result"])
@pytest.mark.parametrize("value,error", [
    (None, TypeError), (1, TypeError), (b"OK", TypeError),
    ("", ValueError), ("PASS", ValueError), (" 未 标记 ", ValueError),
])
def test_invalid_status_is_diagnostic_and_batch_atomic(field, value, error):
    assert_invalid_record_is_atomic(field, value, error)


@pytest.mark.parametrize("field,value", [
    ("forward", "未标记"), ("reverse", "未标记"), ("result", "--"),
])
def test_direction_and_final_status_domains_differ(field, value):
    assert_invalid_record_is_atomic(field, value, ValueError)


@pytest.mark.parametrize("forward,reverse,result,expected", [
    (" ok ", " nG ", " ng ", ("OK", "NG", "NG")),
    (" -- ", "OK", " 未标记 ", ("--", "OK", "未标记")),
    ("NG", "ng", "oK", ("NG", "NG", "OK")),
])
def test_statuses_normalize_without_inferring_final_result(forward, reverse, result, expected):
    day = date(2026, 11, 4)
    records = make_daily_records(1, day)
    records[0].update(forward=forward, reverse=reverse, result=result)
    store = DailyRecordStore()
    store.replace(records, day)
    row = store.records[0]
    assert (row.forward, row.reverse, row.result) == expected
    assert store.summary.ok == (expected[2] == "OK")
    assert store.summary.ng == (expected[2] == "NG")
    assert store.summary.unmarked == (expected[2] == "未标记")


@pytest.mark.parametrize("filters,expected_codes", [
    ({}, ["Straße", "abc-OTHER", "AbC[.+]", " abc[.+] "]),
    ({"station_id": 1}, ["AbC[.+]", " abc[.+] "]),
    ({"result": "OK"}, ["Straße", "AbC[.+]", " abc[.+] "]),
    ({"query": "  ABC  "}, ["abc-OTHER", "AbC[.+]", " abc[.+] "]),
    ({"query": "[.+]"}, ["AbC[.+]", " abc[.+] "]),
    ({"query": "STRASSE"}, ["Straße"]),
    ({"station_id": 1, "result": "OK", "query": "ABC[.+]"}, ["AbC[.+]", " abc[.+] "]),
    ({"station_id": 2, "result": "OK", "query": "abc"}, []),
    ({"query": "OK"}, []),
    ({"query": "  "}, ["Straße", "abc-OTHER", "AbC[.+]", " abc[.+] "]),
])
def test_filters_combine_and_search_only_literal_product_code(filters, expected_codes):
    day = date(2026, 11, 4)
    records = make_daily_records(4, day)
    for row, code, station, result in zip(
        records, [" abc[.+] ", "AbC[.+]", "abc-OTHER", "Straße"],
        [1, 1, 2, 3], ["OK", "OK", "NG", "OK"],
    ):
        row.update(product_code=code, station_id=station, result=result)
    store = DailyRecordStore()
    store.replace(records, day)
    before = (store.records, store.summary, store.station_ids)
    selected = store.select_page(**filters)
    assert [row.product_code for row in selected.rows] == expected_codes
    assert selected.matched_count == len(expected_codes)
    assert (selected.page, selected.page_count) == (1, 1)
    assert (store.records, store.summary, store.station_ids) == before


def test_snapshots_are_immutable_detached_and_keep_stable_duplicate_records():
    day = date(2026, 11, 4)
    records = make_daily_records(3, day)
    for row in records:
        row.update(completed_at=records[0]["completed_at"], station_id=7)
    records[2] = dict(records[0])
    store = DailyRecordStore()
    store.replace(records, day)
    old_records, old_summary = store.records, store.summary
    selected = store.select_page()
    assert [row.product_code for row in old_records] == ["CASE-001", "CASE-002", "CASE-001"]
    assert isinstance(old_records, tuple)
    assert isinstance(selected.rows, tuple)
    with pytest.raises(FrozenInstanceError):
        old_records[0].product_code = "changed"
    with pytest.raises(FrozenInstanceError):
        old_summary.total = 99
    with pytest.raises(FrozenInstanceError):
        selected.page = 99
    records[0]["product_code"] = "changed"
    records[1].clear()
    records.clear()
    assert store.records == old_records
    assert store.records[0].product_code == "CASE-001"
    store.replace([], date(2026, 11, 3))
    assert store.record_date == date(2026, 11, 3)
    assert store.records == ()
    assert store.station_ids == ()
    assert (store.summary.total, store.summary.ok, store.summary.ng,
            store.summary.unmarked, store.summary.pass_rate) == (0, 0, 0, 0, "--")
    assert len(selected.rows) == 3
    assert old_summary.total == 3
    assert len(old_records) == 3


@pytest.mark.parametrize("count", [0, 1, 19, 20, 21, 39, 40, 41])
@pytest.mark.parametrize("requested_page", [-2, 0, 1, 2, 3, 99])
def test_pagination_boundaries_have_exact_rows_and_clamp_page(count, requested_page):
    day = date(2026, 11, 4)
    store = DailyRecordStore()
    store.replace(make_daily_records(count, day), day)
    selected = store.select_page(page=requested_page)
    expected_pages = max(1, (count + 19) // 20)
    expected_page = min(max(1, requested_page), expected_pages)
    expected_codes = [f"CASE-{i:03d}" for i in range(count, 0, -1)]
    start = (expected_page - 1) * 20
    assert selected.matched_count == count
    assert (selected.page, selected.page_count) == (expected_page, expected_pages)
    assert [row.product_code for row in selected.rows] == expected_codes[start:start + 20]
    assert len(selected.rows) <= 20


def test_filter_before_pagination_preserves_order_and_daily_summary():
    day = date(2026, 11, 4)
    store = DailyRecordStore()
    records = make_daily_records(125, day)
    store.replace(records, day)
    before = (store.records, store.summary, store.station_ids)
    selected = store.select_page(station_id=1, result="OK", query=" case- ", page=3)
    assert (selected.matched_count, selected.page, selected.page_count) == (42, 3, 3)
    assert [row.product_code for row in selected.rows] == ["CASE-004", "CASE-001"]
    no_match = store.select_page(result="NG", query="CASE-001", page=3)
    assert (no_match.rows, no_match.matched_count, no_match.page, no_match.page_count) == ((), 0, 1, 1)
    assert (store.records, store.summary, store.station_ids) == before
    assert store.select_page(result="未标记").matched_count == 41


def test_failed_replacement_keeps_previous_page_query_and_other_instances_isolated():
    day = date(2026, 11, 4)
    store, other = DailyRecordStore(), DailyRecordStore()
    store.replace(make_daily_records(65, day), day)
    selected = store.select_page(station_id=1, result="OK", query=" case- ", page=2)
    bad_records = make_daily_records(3, day)
    bad_records[-1]["result"] = "invalid"
    with pytest.raises(ValueError, match=r"records\[2\].result"):
        store.replace(bad_records, day)
    assert store.select_page(station_id=1, result="OK", query=" case- ", page=2) == selected
    assert other.records == ()
    assert other.summary.total == 0
    other.replace(make_daily_records(1, day), day)
    assert store.summary.total == 65
