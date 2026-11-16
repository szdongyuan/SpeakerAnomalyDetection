"""Saved-WAV identity ledger, independent of round completion and Qt."""

from datetime import datetime, timedelta

import pytest

from base.workbench_wav_state import WorkbenchWavState


def test_saved_wav_is_unmarked_and_first_terminal_wins(tmp_path):
    state = WorkbenchWavState()
    when = datetime(2026, 11, 16, 12)
    stations = (("station-a", 1, str(tmp_path / "same.wav")),)
    assert state.register_recording("r1", when, "正转", stations)
    assert not state.register_recording("r1", when, "正转", stations)
    assert state.records_for(when.date()) == [dict(
        completed_at=when, station_id=1, file_name="same.wav", condition="正转", result="未标记",
    )]
    assert state.update_result("r1", "station-a", [{"judgement": "OK"}])
    assert not state.update_result("r1", "station-a", [{"judgement": "NG"}])
    assert not state.register_recording("r1", when + timedelta(hours=1), "反转", (("station-a", 9, str(tmp_path / "changed.wav")),))
    assert state.records_for(when.date())[0]["result"] == "OK"


def test_identity_keeps_same_names_different_requests_paths_and_stations(tmp_path):
    state = WorkbenchWavState()
    when = datetime(2026, 11, 16, 12)
    assert state.register_recording("r1", when, "正转", [
        ("source-a", 3, str(tmp_path / "a" / "same.wav")),
        ("source-b", 1, str(tmp_path / "b" / "same.wav")),
    ])
    assert state.register_recording("r2", when, "反转", [
        ("source-a", 3, str(tmp_path / "c" / "same.wav")),
    ])
    rows = state.records_for(when.date())
    assert len(rows) == 3
    assert [row["station_id"] for row in rows] == [3, 1, 3]
    assert [row["file_name"] for row in rows] == ["same.wav"] * 3
    assert [row["condition"] for row in rows] == ["正转", "正转", "反转"]
    assert state.update_result("r2", "source-a", [{"judgement": "NG"}])
    assert [row["result"] for row in state.records_for(when.date())] == ["未标记", "未标记", "NG"]


def test_partial_duplicate_registration_adds_only_new_identity(tmp_path):
    state = WorkbenchWavState()
    when = datetime(2026, 11, 16, 12)
    a = ("a", 1, str(tmp_path / "a.wav"))
    b = ("b", 2, str(tmp_path / "b.wav"))
    assert state.register_recording("r", when, "正转", [a])
    assert state.update_result("r", "a", [{"judgement": "OK"}])
    assert state.register_recording("r", when, "正转", [a, b])
    assert [r["result"] for r in state.records_for(when.date())] == ["OK", "未标记"]
    assert not state.register_recording("empty", when, "正转", [])


@pytest.mark.parametrize("items,expected", [
    ([{"judgement": "OK"}, {"judgement": "NG"}, {}], "NG"),
    ([{}, {"judgement": " ok "}, {"judgement": None}], "OK"),
    ([{"judgement": " ng "}], "NG"),
    ([], "未标记"), ([{}, {"judgement": "--"}, {"judgement": "unknown"}], "未标记"),
])
def test_shared_aggregation_and_unmarked_terminal_is_still_first_terminal(tmp_path, items, expected):
    state = WorkbenchWavState()
    when = datetime(2026, 11, 16, 12)
    state.register_recording("r", when, "正转", [("a", 1, str(tmp_path / "a.wav"))])
    assert state.update_result("r", "a", items)
    assert not state.update_result("r", "a", [{"judgement": "NG"}])
    assert state.records_for(when.date())[0]["result"] == expected


def test_unknown_identity_does_not_create_or_reserve_terminal(tmp_path):
    state = WorkbenchWavState()
    when = datetime(2026, 11, 16, 12)
    assert not state.update_result("r", "a", [{"judgement": "NG"}])
    assert state.records_for(when.date()) == []
    state.register_recording("r", when, "正转", [("a", 1, str(tmp_path / "a.wav"))])
    assert not state.update_result("r", "other", [{"judgement": "NG"}])
    assert not state.update_result("other", "a", [{"judgement": "NG"}])
    assert state.update_result("r", "a", [{"judgement": "OK"}])


def test_frozen_time_order_days_detached_exports_and_instance_isolation(tmp_path):
    state, other = WorkbenchWavState(), WorkbenchWavState()
    when = datetime(2026, 11, 16, 23, 59, 59)
    later = when + timedelta(seconds=2)
    state.register_recording("r1", when, "正转", [("a", 1, str(tmp_path / "first.wav"))])
    state.register_recording("r2", when, "正转", [("a", 1, str(tmp_path / "second.wav"))])
    state.register_recording("r3", later, "反转", [("a", 1, str(tmp_path / "third.wav"))])
    exported = state.records_for(when.date())
    exported[0]["file_name"] = "changed.wav"
    exported[1].clear()
    exported.clear()
    assert state.update_result("r1", "a", [{"judgement": "OK"}])
    assert [r["file_name"] for r in state.records_for(when.date())] == ["first.wav", "second.wav"]
    assert [r["completed_at"] for r in state.records_for(when.date())] == [when, when]
    assert [r["completed_at"] for r in state.records_for(later.date())] == [later]
    assert other.records_for(when.date()) == []
    assert not other.update_result("r1", "a", [{"judgement": "NG"}])
