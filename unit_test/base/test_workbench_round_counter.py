from datetime import date
from pathlib import Path

import pytest

import base.workbench_round_store as round_store_module
from base.workbench_round_counter import WorkbenchRoundCounter
from base.workbench_round_store import (
    RoundSnapshot,
    WorkbenchRoundStateError,
    WorkbenchRoundStore,
)


@pytest.fixture
def store(tmp_path):
    return WorkbenchRoundStore(tmp_path / "round.json")


def assert_committed(counter, store, day, number):
    snapshot = RoundSnapshot(day, number)
    assert counter.current_snapshot == snapshot
    assert store.load() == snapshot
    assert counter.can_record
    return snapshot


def test_missing_state_is_initialized_only_on_explicit_refresh(store):
    counter = WorkbenchRoundCounter(store)
    day = date(2026, 11, 12)
    assert counter.current_snapshot is None
    assert not counter.can_record
    assert store.load() is None

    assert counter.refresh(day) == RoundSnapshot(day, 1)
    assert_committed(counter, store, day, 1)


@pytest.mark.parametrize("saved_day", [date(2026, 11, 12), date(2026, 11, 11)])
def test_load_restores_same_day_or_resets_other_day(store, saved_day):
    day = date(2026, 11, 12)
    store.save(RoundSnapshot(saved_day, 10000))
    counter = WorkbenchRoundCounter(store)

    expected_number = 10000 if saved_day == day else 1
    assert counter.prepare_new_round(day) == RoundSnapshot(day, expected_number)
    assert_committed(counter, store, day, expected_number)


def test_corrupt_state_is_never_silently_replaced(store):
    store.path.write_bytes(b"invalid JSON")
    counter = WorkbenchRoundCounter(store)

    for operation in (counter.refresh, counter.prepare_new_round):
        with pytest.raises(WorkbenchRoundStateError, match="load"):
            operation(date(2026, 11, 12))
        assert counter.current_snapshot is None
        assert not counter.can_record
        assert store.path.read_bytes() == b"invalid JSON"


def test_initial_save_failure_remains_unavailable_until_retry(store, monkeypatch):
    counter = WorkbenchRoundCounter(store)
    day = date(2026, 11, 12)
    real_save = store.save
    attempts = []

    def fail_once(snapshot):
        attempts.append(snapshot)
        if len(attempts) == 1:
            raise WorkbenchRoundStateError("injected save failure")
        real_save(snapshot)

    monkeypatch.setattr(store, "save", fail_once)
    with pytest.raises(WorkbenchRoundStateError):
        counter.prepare_new_round(day)
    assert counter.current_snapshot is None
    assert not counter.can_record
    assert store.load() is None

    assert counter.prepare_new_round(day) == RoundSnapshot(day, 1)
    assert attempts == [RoundSnapshot(day, 1)] * 2
    assert_committed(counter, store, day, 1)


def test_early_next_round_advances_once_and_old_results_do_not_advance_new_round(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    keys = {("slow", "left"), ("fast", "right")}
    counter.open_round("A", counter.prepare_new_round(day), keys)
    counter.recordings_completed("A", day)
    counter.analysis_completed("A", ("slow", "left"), day)
    assert_committed(counter, store, day, 1)

    snapshot = counter.prepare_new_round(day)
    assert snapshot == assert_committed(counter, store, day, 2)
    # A real bridge may reject startup. Preparing again must reuse the number.
    assert counter.prepare_new_round(day) == snapshot
    counter.open_round("B", snapshot, keys)
    counter.analysis_completed("A", ("fast", "right"), day)
    counter.recordings_completed("A", day)
    assert_committed(counter, store, day, 2)

    counter.analysis_completed("B", ("fast", "right"), day)
    counter.recordings_completed("B", day)
    assert_committed(counter, store, day, 2)
    counter.analysis_completed("B", ("slow", "left"), day)
    assert_committed(counter, store, day, 3)
    counter.analysis_completed("B", ("slow", "left"), day)
    assert counter.prepare_new_round(day) == assert_committed(counter, store, day, 3)


def test_automatic_advance_waits_for_all_expected_pairs_and_recordings(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    keys = {("first", "left"), ("last", "left"), ("last", "right")}
    snapshot = counter.prepare_new_round(day)
    counter.open_round("A", snapshot, keys)
    # The input is frozen, and a duplicate open is idempotent.
    counter.open_round("A", snapshot, keys)
    keys.clear()
    counter.analysis_completed("A", ("last", "right"), day)
    counter.analysis_completed("A", ("last", "left"), day)
    counter.analysis_completed("A", ("first", "right"), day)  # Not applicable.
    counter.analysis_completed("unknown", ("first", "left"), day)
    assert_committed(counter, store, day, 1)
    counter.recordings_completed("A", day)
    counter.analysis_completed("A", ("last", "right"), day)  # Duplicate.
    assert_committed(counter, store, day, 1)

    counter.analysis_completed("A", ("first", "left"), day)
    assert_committed(counter, store, day, 2)
    assert counter.prepare_new_round(day) == RoundSnapshot(day, 2)
    counter.recordings_completed("A", day)
    assert_committed(counter, store, day, 2)


@pytest.mark.parametrize("keys", [set(), {("condition", "station")}])
def test_results_before_recording_boundary_do_not_advance_early(store, keys):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    counter.open_round("A", counter.prepare_new_round(day), keys)
    for key in keys:
        counter.analysis_completed("A", key, day)
    assert_committed(counter, store, day, 1)

    counter.recordings_completed("A", day)
    assert_committed(counter, store, day, 2)


def test_unfinished_round_must_be_completed_or_explicitly_abandoned(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    snapshot = counter.prepare_new_round(day)
    counter.open_round("A", snapshot, set())

    with pytest.raises(ValueError, match="unfinished"):
        counter.prepare_new_round(day)
    with pytest.raises(ValueError, match="unfinished"):
        counter.open_round("B", snapshot, set())
    assert_committed(counter, store, day, 1)

    counter.abandon_round("A", day)
    counter.recordings_completed("A", day)
    assert counter.prepare_new_round(day) == snapshot
    counter.open_round("B", snapshot, set())
    counter.recordings_completed("B", day)
    assert_committed(counter, store, day, 2)


def test_partial_round_abandonment_ignores_late_results_without_consuming_number(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    snapshot = counter.prepare_new_round(day)
    counter.open_round("old-product", snapshot, {key})
    counter.abandon_round("old-product", day)
    counter.open_round("new-product", counter.prepare_new_round(day), {key})
    counter.analysis_completed("old-product", key, day)
    counter.recordings_completed("old-product", day)
    assert_committed(counter, store, day, 1)
    counter.recordings_completed("new-product", day)
    counter.analysis_completed("new-product", key, day)
    assert_committed(counter, store, day, 2)


def test_unfinished_old_day_round_is_protected_until_last_recording(store):
    old_day, new_day = date(2026, 11, 12), date(2026, 11, 13)
    store.save(RoundSnapshot(old_day, 8))
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    frozen = counter.prepare_new_round(old_day)
    counter.open_round("A", frozen, {key})

    assert counter.refresh(new_day) == frozen
    assert_committed(counter, store, old_day, 8)
    counter.analysis_completed("A", key, new_day)
    assert_committed(counter, store, old_day, 8)
    counter.recordings_completed("A", new_day)
    assert_committed(counter, store, new_day, 1)
    assert counter.prepare_new_round(new_day) == RoundSnapshot(new_day, 1)
    counter.analysis_completed("A", key, new_day)
    assert_committed(counter, store, new_day, 1)


@pytest.mark.parametrize("transition", ["refresh", "prepare", "old_analysis"])
def test_finished_old_day_round_resets_without_carrying_advance_into_new_day(store, transition):
    old_day, new_day = date(2026, 11, 12), date(2026, 11, 13)
    store.save(RoundSnapshot(old_day, 8))
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    counter.open_round("A", counter.prepare_new_round(old_day), {key})
    counter.recordings_completed("A", old_day)
    assert_committed(counter, store, old_day, 8)

    if transition == "refresh":
        counter.refresh(new_day)
    elif transition == "prepare":
        counter.prepare_new_round(new_day)
    else:
        counter.analysis_completed("A", key, new_day)
    assert_committed(counter, store, new_day, 1)
    counter.open_round("B", counter.prepare_new_round(new_day), {key})
    counter.analysis_completed("A", key, new_day)
    assert_committed(counter, store, new_day, 1)
    counter.recordings_completed("B", new_day)
    counter.analysis_completed("B", key, new_day)
    assert_committed(counter, store, new_day, 2)


def test_abandoning_unfinished_old_day_round_allows_new_day_reset(store):
    old_day, new_day = date(2026, 11, 12), date(2026, 11, 13)
    store.save(RoundSnapshot(old_day, 8))
    counter = WorkbenchRoundCounter(store)
    counter.open_round("A", counter.prepare_new_round(old_day), set())
    counter.refresh(new_day)
    assert_committed(counter, store, old_day, 8)

    counter.abandon_round("A", new_day)
    assert_committed(counter, store, new_day, 1)
    counter.recordings_completed("A", new_day)
    assert counter.prepare_new_round(new_day) == RoundSnapshot(new_day, 1)


def test_idle_date_reset_and_repeated_refresh_do_not_increment_or_rewrite(store, monkeypatch):
    old_day, new_day = date(2026, 11, 12), date(2026, 11, 13)
    store.save(RoundSnapshot(old_day, 8))
    counter = WorkbenchRoundCounter(store)
    counter.refresh(old_day)
    real_save = store.save
    saved = []

    def track_save(snapshot):
        saved.append(snapshot)
        real_save(snapshot)

    monkeypatch.setattr(store, "save", track_save)
    counter.refresh(new_day)
    counter.refresh(new_day)
    counter.prepare_new_round(new_day)
    assert saved == [RoundSnapshot(new_day, 1)]
    assert_committed(counter, store, new_day, 1)


@pytest.mark.parametrize("finish_recordings", [False, True])
def test_restart_restores_only_committed_number_not_old_round_or_progress(store, finish_recordings):
    day = date(2026, 11, 12)
    store.save(RoundSnapshot(day, 8))
    key = ("condition", "station")
    first = WorkbenchRoundCounter(store)
    first.open_round("old-process", first.prepare_new_round(day), {key})
    if finish_recordings:
        first.recordings_completed("old-process", day)
    assert_committed(first, store, day, 8)

    restarted = WorkbenchRoundCounter(WorkbenchRoundStore(store.path))
    restarted.refresh(day)
    restarted.analysis_completed("old-process", key, day)
    restarted.recordings_completed("old-process", day)
    assert_committed(restarted, store, day, 8)
    restarted.open_round("new-process", restarted.prepare_new_round(day), set())
    restarted.recordings_completed("new-process", day)
    assert_committed(restarted, store, day, 9)


@pytest.mark.parametrize("trigger", ["analysis", "prepare", "recordings"])
@pytest.mark.parametrize("retry", ["refresh", "prepare", "duplicate"])
def test_failed_advance_retries_exact_target_once(store, monkeypatch, trigger, retry):
    day = date(2026, 11, 12)
    store.save(RoundSnapshot(day, 8))
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    old = counter.prepare_new_round(day)
    counter.open_round("A", old, {key})
    if trigger == "recordings":
        counter.analysis_completed("A", key, day)
    else:
        counter.recordings_completed("A", day)
    original_bytes = store.path.read_bytes()
    real_replace = round_store_module.os.replace
    attempts = []

    def fail_once(source, destination):
        attempts.append(Path(source).read_text(encoding="utf-8"))
        if len(attempts) == 1:
            raise PermissionError("injected disk replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(round_store_module.os, "replace", fail_once)
    with pytest.raises(WorkbenchRoundStateError, match="save"):
        if trigger == "analysis":
            counter.analysis_completed("A", key, day)
        elif trigger == "prepare":
            counter.prepare_new_round(day)
        else:
            counter.recordings_completed("A", day)
    assert counter.current_snapshot == old
    assert store.path.read_bytes() == original_bytes
    assert not counter.can_record
    with pytest.raises(ValueError, match="uncommitted|unfinished"):
        counter.open_round("unpersisted", RoundSnapshot(day, 9), set())

    if retry == "refresh":
        counter.refresh(day)
    elif retry == "prepare":
        assert counter.prepare_new_round(day) == RoundSnapshot(day, 9)
    elif trigger == "analysis":
        counter.analysis_completed("A", key, day)
    elif trigger == "recordings":
        counter.recordings_completed("A", day)
    else:
        counter.prepare_new_round(day)
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert_committed(counter, store, day, 9)
    counter.analysis_completed("A", key, day)
    counter.recordings_completed("A", day)
    assert counter.prepare_new_round(day) == RoundSnapshot(day, 9)
    assert len(attempts) == 2
    assert_committed(counter, store, day, 9)


def test_counters_own_separate_round_contexts(store):
    day = date(2026, 11, 12)
    first = WorkbenchRoundCounter(store)
    second = WorkbenchRoundCounter(WorkbenchRoundStore(store.path.parent / "second.json"))
    first.open_round("same-id", first.prepare_new_round(day), set())
    second.open_round("same-id", second.prepare_new_round(day), set())
    first.recordings_completed("same-id", day)
    assert_committed(first, store, day, 2)
    assert second.current_snapshot == RoundSnapshot(day, 1)


def test_completed_waiting_round_is_not_abandoned_by_product_switch(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    counter.open_round("A", counter.prepare_new_round(day), {key})
    counter.recordings_completed("A", day)
    counter.abandon_round("A", day)
    counter.analysis_completed("A", key, day)
    assert_committed(counter, store, day, 2)


def test_failed_daily_reset_retries_same_target_and_retires_old_round(store, monkeypatch):
    old_day, new_day = date(2026, 11, 12), date(2026, 11, 13)
    old = RoundSnapshot(old_day, 8)
    store.save(old)
    counter = WorkbenchRoundCounter(store)
    key = ("condition", "station")
    counter.open_round("A", counter.prepare_new_round(old_day), {key})
    counter.recordings_completed("A", old_day)
    real_save = store.save
    attempts = []

    def fail_twice(snapshot):
        attempts.append(snapshot)
        if len(attempts) <= 2:
            raise WorkbenchRoundStateError("injected persistent failure")
        real_save(snapshot)

    monkeypatch.setattr(store, "save", fail_twice)
    with pytest.raises(WorkbenchRoundStateError):
        counter.refresh(new_day)
    assert counter.current_snapshot == store.load() == old
    assert not counter.can_record
    with pytest.raises(WorkbenchRoundStateError):
        counter.prepare_new_round(new_day)
    assert counter.current_snapshot == store.load() == old
    assert not counter.can_record

    counter.refresh(new_day)
    assert attempts == [RoundSnapshot(new_day, 1)] * 3
    assert_committed(counter, store, new_day, 1)
    counter.analysis_completed("A", key, new_day)
    assert counter.prepare_new_round(new_day) == RoundSnapshot(new_day, 1)
    assert_committed(counter, store, new_day, 1)


def test_frozen_identity_cannot_be_replaced_and_stale_snapshot_cannot_be_opened(store):
    day = date(2026, 11, 12)
    counter = WorkbenchRoundCounter(store)
    snapshot = counter.prepare_new_round(day)
    counter.open_round("A", snapshot, set())
    with pytest.raises(ValueError, match="frozen"):
        counter.open_round("A", snapshot, {("new", "task")})
    with pytest.raises(ValueError, match="frozen"):
        counter.open_round("A", RoundSnapshot(day, 2), set())
    counter.recordings_completed("A", day)
    with pytest.raises(ValueError, match="uncommitted"):
        counter.open_round("B", snapshot, set())
    assert_committed(counter, store, day, 2)
