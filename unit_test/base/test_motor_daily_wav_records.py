"""Independent saved-WAV snapshot validation, filtering and pagination."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from base.motor_daily_wav_records import DailyWavRecord, DailyWavRecordStore
from unit_test.daily_record_cases import make_daily_records, make_daily_wav_records


@pytest.fixture
def day():
    return date(2026, 11, 16)


def snapshot(store):
    return store.record_date, store.records, store.summary, store.station_ids


def test_initial_state_and_default_local_date(monkeypatch):
    class LocalDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 16)

    monkeypatch.setattr("base.motor_daily_wav_records.date", LocalDate)
    store = DailyWavRecordStore()
    assert snapshot(store) == (LocalDate.today(), (), store.summary, ())
    assert vars(store.summary) == dict(incomplete=0, total=0, ok=0, ng=0, unmarked=0, pass_rate="--")
    store.replace(make_daily_wav_records(1, LocalDate.today()))
    assert store.record_date == LocalDate.today()


def test_replace_counts_single_file_result_and_sorts_full_precision(day):
    rows = make_daily_wav_records(3, day)
    rows[0].update(result=" ok ", station_id=9, extra="ignored")
    store = DailyWavRecordStore()
    assert store.replace(tuple(rows), day) is None
    assert [r.file_name for r in store.records] == ["CASE-003.wav", "CASE-002.wav", "CASE-001.wav"]
    assert store.station_ids == (2, 3, 9)
    assert vars(store.summary) == dict(incomplete=0, total=3, ok=1, ng=1, unmarked=1, pass_rate="33.3%")
    assert vars(store.records[-1]) == {key: value for key, value in rows[0].items() if key != "extra"} | {"result": "OK", "analysis_reference": None}
    assert set(make_daily_records(1, day)[0]) == {
        "completed_at", "station_id", "product_code", "forward", "reverse", "result",
    }


@pytest.mark.parametrize("batch", [None, "rows", {}, iter(())])
def test_invalid_container_preserves_snapshot(day, batch):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(1, day), day)
    before = snapshot(store)
    with pytest.raises(TypeError, match="records"):
        store.replace(batch, day)
    assert snapshot(store) == before


@pytest.mark.parametrize("bad_day", ["2026-11-16", datetime(2026, 11, 16), 1])
def test_invalid_date_preserves_snapshot(day, bad_day):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(1, day), day)
    before = snapshot(store)
    with pytest.raises(TypeError, match="record_date"):
        store.replace([], bad_day)
    assert snapshot(store) == before


@pytest.mark.parametrize("item", [None, [], "row", 1])
def test_invalid_item_reports_index_and_preserves_snapshot(day, item):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(1, day), day)
    before = snapshot(store)
    with pytest.raises(TypeError, match=r"records\[1\]"):
        store.replace([make_daily_wav_records(1, day)[0], item], day)
    assert snapshot(store) == before


@pytest.mark.parametrize("field", ["completed_at", "station_id", "file_name", "condition", "result"])
def test_missing_field_is_atomic_and_diagnostic(day, field):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(1, day), day)
    before = snapshot(store)
    rows = make_daily_wav_records(2, day)
    del rows[1][field]
    with pytest.raises(ValueError, match=rf"records\[1\].{field}"):
        store.replace(rows, day)
    assert snapshot(store) == before


@pytest.mark.parametrize("field,value,error", [
    ("completed_at", date(2026, 11, 16), TypeError),
    ("completed_at", "12:00:00", TypeError),
    ("completed_at", None, TypeError),
    ("completed_at", datetime(2026, 11, 16, tzinfo=timezone.utc), ValueError),
    ("completed_at", datetime(2026, 11, 15), ValueError),
    ("station_id", True, TypeError), ("station_id", False, TypeError),
    ("station_id", 1.0, TypeError), ("station_id", "1", TypeError),
    ("station_id", None, TypeError), ("station_id", 0, ValueError),
    ("station_id", -1, ValueError),
    ("file_name", None, TypeError), ("file_name", 123, TypeError),
    ("file_name", b"a.wav", TypeError), ("file_name", "", ValueError),
    ("file_name", "  ", ValueError),
    ("condition", None, TypeError), ("condition", 1, TypeError),
    ("condition", "  ", ValueError), ("condition", "\t\n", ValueError),
    ("condition", "", ValueError),
    ("result", None, TypeError), ("result", 1, TypeError),
    ("result", b"OK", TypeError), ("result", "", ValueError),
    ("result", "--", ValueError), ("result", "PASS", ValueError),
])
def test_invalid_values_are_atomic_and_diagnostic(day, field, value, error):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(3, day), day)
    before = snapshot(store)
    rows = make_daily_wav_records(2, day)
    rows[1][field] = value
    with pytest.raises(error, match=rf"records\[1\].{field}"):
        store.replace(rows, day)
    assert snapshot(store) == before


@pytest.mark.parametrize("status,expected", [(" ok ", "OK"), (" nG ", "NG"), (" 未标记 ", "未标记")])
def test_result_normalization(day, status, expected):
    rows = make_daily_wav_records(1, day)
    rows[0]["result"] = status
    store = DailyWavRecordStore()
    store.replace(rows, day)
    assert store.records[0].result == expected


@pytest.mark.parametrize("condition", ["正转", "反转", "负载升速", "forward", "  温升 复测  "])
def test_nonempty_condition_names_are_preserved_exactly(day, condition):
    rows = make_daily_wav_records(1, day)
    rows[0]["condition"] = condition
    store = DailyWavRecordStore()
    store.replace(rows, day)
    assert store.records[0].condition == condition


def test_snapshot_is_detached_immutable_and_stable_for_ties(day):
    rows = make_daily_wav_records(3, day)
    for row in rows:
        row["completed_at"] = rows[0]["completed_at"]
    rows[2] = dict(rows[0])
    store = DailyWavRecordStore()
    store.replace(rows, day)
    before = snapshot(store)
    selected = store.select_page()
    assert [r.file_name for r in store.records] == ["CASE-001.wav", "CASE-002.wav", "CASE-001.wav"]
    for obj, field in [(store.records[0], "file_name"), (store.summary, "total"), (selected, "page")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, "changed")
    rows[0]["file_name"] = "changed.wav"
    rows[1].clear()
    rows.clear()
    assert snapshot(store) == before
    store.replace([], date(2026, 11, 15))
    assert store.record_date == date(2026, 11, 15)
    assert store.records == store.station_ids == ()
    assert vars(store.summary) == dict(incomplete=0, total=0, ok=0, ng=0, unmarked=0, pass_rate="--")
    assert len(selected.rows) == 3


@pytest.mark.parametrize("filters,expected", [
    ({}, ["Straße.wav", "abc-OTHER.wav", "AbC[.+].wav", " abc[.+].wav "]),
    ({"query": "  ABC  "}, ["abc-OTHER.wav", "AbC[.+].wav", " abc[.+].wav "]),
    ({"query": "[.+]"}, ["AbC[.+].wav", " abc[.+].wav "]),
    ({"query": "STRASSE"}, ["Straße.wav"]),
    ({"station_id": 1, "result": "OK", "query": "ABC[.+]"}, ["AbC[.+].wav", " abc[.+].wav "]),
    ({"station_id": 2, "result": "OK", "query": "abc"}, []),
    ({"query": "正转"}, []), ({"query": "OK"}, []),
    ({"query": "  "}, ["Straße.wav", "abc-OTHER.wav", "AbC[.+].wav", " abc[.+].wav "]),
])
def test_filename_search_is_literal_casefolded_and_combines_filters(day, filters, expected):
    rows = make_daily_wav_records(4, day)
    for row, name, station, result in zip(rows,
        [" abc[.+].wav ", "AbC[.+].wav", "abc-OTHER.wav", "Straße.wav"],
        [1, 1, 2, 3], ["OK", "OK", "NG", "OK"],
    ):
        row.update(file_name=name, station_id=station, result=result)
    store = DailyWavRecordStore()
    store.replace(rows, day)
    before = snapshot(store)
    page = store.select_page(**filters)
    assert [r.file_name for r in page.rows] == expected
    assert (page.matched_count, page.page, page.page_count) == (len(expected), 1, 1)
    assert snapshot(store) == before


@pytest.mark.parametrize("count", [0, 1, 20, 21, 39, 40, 41])
@pytest.mark.parametrize("requested", [-2, 0, 1, 2, 3, 99])
def test_pagination_exact_rows_and_clamped_page(day, count, requested):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(count, day), day)
    page = store.select_page(page=requested)
    pages = max(1, (count + 19) // 20)
    current = min(max(1, requested), pages)
    names = [f"CASE-{index:03d}.wav" for index in range(count, 0, -1)]
    assert (page.matched_count, page.page, page.page_count) == (count, current, pages)
    start = (current - 1) * 20
    assert [r.file_name for r in page.rows] == names[start:start + 20]


def test_filter_before_pagination_and_instance_isolation(day):
    store, other = DailyWavRecordStore(), DailyWavRecordStore()
    store.replace(make_daily_wav_records(125, day), day)
    before = snapshot(store)
    page = store.select_page(station_id=1, result="OK", query=" case- ", page=3)
    assert (page.matched_count, page.page, page.page_count) == (42, 3, 3)
    assert [r.file_name for r in page.rows] == ["CASE-004.wav", "CASE-001.wav"]
    assert snapshot(store) == before
    assert other.records == ()
    other.replace(make_daily_wav_records(1, day), day)
    assert store.summary.total == 125


def test_old_display_fields_still_construct_record(day):
    record = DailyWavRecord(datetime(2026, 11, 16), 9, "record.wav", "正转", "OK")
    assert record.analysis_reference is None


def test_reference_survives_snapshot_sorting_paging_and_caller_mutation(day, tmp_path):
    from base.record_analysis_reference import RecordAnalysisReference
    from base.workbench_recording_protocol import StationWavChannel

    reference = RecordAnalysisReference("request", "source", "condition", str(tmp_path / "record.wav"),
        [StationWavChannel("mic", "Microphone", "sound", "device", 4, 0)])
    rows = make_daily_wav_records(3, day)
    rows[0]["analysis_reference"] = reference
    store, other = DailyWavRecordStore(), DailyWavRecordStore()
    store.replace(rows, day)
    before = snapshot(store)
    rows[0]["analysis_reference"] = None
    rows.clear()
    assert store.records[-1].analysis_reference is reference
    assert store.select_page().rows[-1].analysis_reference is reference
    assert other.records == ()
    assert vars(store.summary) == dict(incomplete=0, total=3, ok=1, ng=1, unmarked=1, pass_rate="33.3%")
    assert snapshot(store) == before


@pytest.mark.parametrize("value", [{}, "request", 1, object()])
def test_invalid_reference_preserves_entire_snapshot(day, value):
    store = DailyWavRecordStore()
    store.replace(make_daily_wav_records(3, day), day)
    before = snapshot(store)
    rows = make_daily_wav_records(2, day)
    rows[1]["analysis_reference"] = value
    with pytest.raises(TypeError, match=r"records\[1\].analysis_reference"):
        store.replace(rows, day)
    assert snapshot(store) == before
