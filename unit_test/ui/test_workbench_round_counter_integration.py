"""Daily numbers follow authenticated whole-round recording/analysis boundaries."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from base.workbench_round_counter import WorkbenchRoundCounter
from base.workbench_round_store import RoundSnapshot, WorkbenchRoundStore, WorkbenchRoundStateError
from base.workbench_analysis_input import WorkbenchAnalysisInput
from base.workbench_analysis_config import AnalysisTestQueue
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import AnalysisServiceStub
from unit_test.ui.test_workbench_result_sync import result_event


@pytest.fixture
def rounds(paired_recording_workbench, tmp_path, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    w = paired_recording_workbench
    clock = SimpleNamespace(value=datetime(2026, 11, 12, 12))
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: clock.value))
    store = WorkbenchRoundStore(tmp_path / "round.json")
    counter = WorkbenchRoundCounter(store)
    shown, errors, events = [], [], []
    w.controller.bind_round_counter(counter, shown.append, errors.append)
    service = AnalysisServiceStub()
    w.controller.bind_analysis_service(service, events.append)
    w.controller.activate_analysis()

    def record():
        w.controller.start_recording()
        w.complete()
        return service.submissions[-1]["tasks"]

    def deliver(*tasks):
        service.events.extend(result_event(task) for task in tasks)
        w.controller.poll_analysis_events()

    def assert_number(number, day=None):
        assert shown[-1] == number
        assert store.load() == RoundSnapshot(day or clock.value.date(), number)

    return SimpleNamespace(w=w, service=service, counter=counter, store=store, shown=shown,
                           errors=errors, events=events, clock=clock, record=record,
                           deliver=deliver, assert_number=assert_number)


def assert_wav_number(request, number):
    for station in request.station_group.stations:
        assert f"_M-02_{number}_SN-008.wav" in station.filename


def test_round_freezes_wav_number_and_waits_for_every_station_condition(rounds):
    r = rounds
    r.assert_number(1)
    first = r.record()
    r.w.identity.sample = "edited"
    second = r.record()
    for request in r.w.requests:
        assert_wav_number(request, 1)
    for task in (*first, *second):
        assert "_M-02_1_SN-008.wav" in Path(task["wav_path"]).name
    r.deliver(*second, *first[1:])
    r.assert_number(1)
    r.deliver(first[0])
    r.assert_number(2)
    r.deliver(*first, *second)
    r.assert_number(2)


def test_early_next_round_advances_before_bridge_and_old_results_do_not_advance(rounds):
    r = rounds
    first, second = r.record(), r.record()
    r.w.outcome.on_start = lambda: r.assert_number(2)
    newer_first = r.record()
    r.w.outcome.on_start = None
    assert_wav_number(r.w.requests[-1], 2)
    r.deliver(*first, *second)
    r.assert_number(2)
    newer_second = r.record()
    r.deliver(*newer_second, *newer_first)
    r.assert_number(3)


@pytest.mark.parametrize("terminal", ["failure", "cancelled"])
@pytest.mark.parametrize("condition", ["first", "later"])
def test_recording_retry_keeps_number_and_later_condition_round(rounds, terminal, condition):
    r = rounds
    if condition == "later":
        r.record()
    r.w.controller.start_recording()
    failed_round = r.w.controller._current_round_id
    r.w.complete(outcome=terminal)
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 1)
    assert (r.w.controller._current_round_id == failed_round) == (condition == "later")
    r.w.complete()
    r.assert_number(1)


@pytest.mark.parametrize("change", ["config", "source", "same"])
@pytest.mark.parametrize("refresh_at", ["partial", "inflight", "preparation"])
def test_refresh_abandons_only_changed_partial_round(rounds, monkeypatch, change, refresh_at):
    from ui.sequence import motor_workbench_controller as module
    from ui.sequence.motor_workbench_controller import WorkbenchProductSelectionPorts

    r = rounds
    r.w.controller.bind_product_selection(WorkbenchProductSelectionPorts(
        product_manager=None, show_catalog=lambda *_: None, set_enabled=lambda *_: None,
        is_busy=lambda: False, is_closing=lambda: False, notify_error=lambda *_: None))

    def refresh():
        if change == "config":
            r.w.repository.result = configuration_with(r.w.config, project_name="changed")
        elif change == "source":
            r.w.repository.source_identity = "another.json"
        r.w.controller.refresh_configuration()

    if refresh_at == "preparation":
        build = module.build_recording_request

        def build_and_refresh(*args, **kwargs):
            request = build(*args, **kwargs)
            refresh()
            return request

        with monkeypatch.context() as patch:
            patch.setattr(module, "build_recording_request", build_and_refresh)
            first = r.record()
    else:
        r.w.controller.start_recording()
        if refresh_at == "inflight":
            refresh()
        r.w.complete()
        first = r.service.submissions[-1]["tasks"]
        if refresh_at == "partial":
            refresh()
    r.deliver(*first)
    next_tasks = r.record()
    assert r.w.requests[-1].station_group.condition_id == ("cold" if change == "same" else "warm")
    if change == "same":
        r.deliver(*next_tasks)
        r.assert_number(2)
    else:
        r.deliver(*next_tasks)
        r.assert_number(1)
        last = r.record()
        r.deliver(*last)
        r.assert_number(2)


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled"])
def test_synchronous_terminal_then_reentrant_start_opens_counter_once(rounds, terminal):
    r = rounds
    if terminal == "success":
        r.record()

    def finish_and_restart():
        r.w.outcome.on_start = None
        r.w.complete(outcome=terminal)
        r.w.controller.start_recording()

    r.w.outcome.on_start = finish_and_restart
    r.w.controller.start_recording()
    assert r.w.controller._in_flight is not None
    assert_wav_number(r.w.requests[-1], 2 if terminal == "success" else 1)
    r.assert_number(2 if terminal == "success" else 1)


def test_asymmetric_applicable_stations_wait_only_for_frozen_expected_pairs(rounds):
    r = rounds
    conditions = r.w.config.analysis.context.conditions
    conditions = (replace(conditions[0], applicable_station_ids=("right", "left")),
                  replace(conditions[1], applicable_station_ids=("right",)))
    r.w.repository.result = configuration_with(r.w.config, conditions=conditions)
    r.w.controller.refresh_configuration()
    first, second = r.record(), r.record()
    assert len(first) == 2 and len(second) == 1
    r.deliver(*second, first[0])
    r.assert_number(1)
    r.deliver(first[1])
    r.assert_number(2)


def test_results_before_final_archive_wait_for_recordings_completed(rounds):
    r = rounds
    r.deliver(*r.record())
    r.w.outcome.on_archive = lambda *_: None
    r.w.controller.start_recording()
    r.w.complete()
    completion, _, _, archived = r.w.archive_submissions[-1]
    attempt = r.w.controller._in_flight
    # Exercise an early normal terminal using real task identities and paths;
    # ordinary submission happens after archive, but completion must be order safe.
    r.w.controller._submit_saved_recording(
        WorkbenchAnalysisInput(attempt.analysis_context, completion), attempt.round_context)
    r.deliver(*r.service.submissions[-1]["tasks"])
    r.assert_number(1)
    archived(completion.request_id, None)
    r.assert_number(2)
    archived(completion.request_id, None)
    r.assert_number(2)


@pytest.mark.parametrize("empty", ["all", "last"])
def test_empty_analysis_conditions_do_not_wait_for_nonexistent_events(rounds, empty):
    r = rounds
    conditions = r.w.config.analysis.context.conditions
    conditions = (conditions[0], replace(conditions[1], test_queue="empty"))
    queues = {**r.w.config.analysis.test_queues,
              "empty": AnalysisTestQueue("wav", {"display_sequence": []})}
    if empty == "all":
        conditions = tuple(replace(c, test_queue="empty") for c in conditions)
    r.w.repository.result = configuration_with(
        r.w.config, conditions=conditions, test_queues=queues,
        parameters={**r.w.config.recording.parameters_by_queue, "empty": {"total_time": 2}})
    r.w.controller.refresh_configuration()
    r.w.controller.start_recording()
    r.w.complete()
    if empty == "last":
        r.deliver(*r.service.submissions[-1]["tasks"])
    r.assert_number(1)
    r.w.controller.start_recording()
    r.w.complete()
    assert r.events[-1]["event_type"] == "analysis_skipped"
    r.assert_number(2)


def test_midnight_keeps_remaining_condition_wav_and_old_results_do_not_consume_new_day(rounds):
    r = rounds
    # Establish a number greater than one without reaching into counter state.
    a, b = r.record(), r.record()
    r.deliver(*a, *b)
    r.clock.value = datetime(2026, 11, 12, 23, 59, 59)
    first = r.record()
    r.clock.value = datetime(2026, 11, 13, 0, 0, 3)
    r.w.controller.refresh_recording_state()
    r.assert_number(2, datetime(2026, 11, 12).date())
    second = r.record()
    assert_wav_number(r.w.requests[-1], 2)
    assert Path(second[0]["wav_path"]).name.startswith("2026-11-13_00-00-03")
    assert r.w.archive_submissions[-1][2] == r.clock.value.date()
    r.assert_number(1)
    r.deliver(*first, *second)
    r.assert_number(1)
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 1)


def test_idle_periodic_refresh_resets_new_day_without_analysis_event(rounds):
    r = rounds
    first, second = r.record(), r.record()
    r.deliver(*first, *second)
    r.assert_number(2)
    r.clock.value = datetime(2026, 11, 13)
    r.w.controller.refresh_recording_state()
    r.assert_number(1)


@pytest.mark.parametrize("location,field", [
    ("event", "station_id"), ("event", "condition_id"),
    ("payload", "station_id"), ("payload", "condition_id"), ("payload", "task_id"),
    ("payload", None),
])
def test_malformed_identity_cannot_complete_round(rounds, location, field, caplog):
    r = rounds
    first, second = r.record(), r.record()
    r.deliver(*second, *first[1:])
    incoming = result_event(first[0])
    if field is None:
        incoming["payload"] = []
    else:
        (incoming if location == "event" else incoming["payload"])[field] = "forged"
    r.service.events.append(incoming)
    r.w.controller.poll_analysis_events()
    r.assert_number(1)
    assert first[0]["task_id"] in caplog.text
    r.deliver({**first[0], "task_id": "unknown"})
    r.assert_number(1)


@pytest.mark.parametrize("rejection", ["admission", "bridge"])
def test_rejection_preserves_display_and_prepared_number_for_retry(rounds, workbench, rejection):
    r = rounds
    r.record()
    r.record()
    workbench.panel.show_station_result(1, "OK", "NG", "NG", [["SPL", "10", "20"]])
    before = workbench.panel.get_station_data(1)
    if rejection == "admission":
        r.w.outcome.hardware_available = False
    else:
        r.w.outcome.error = RuntimeError("busy")
    r.w.controller.start_recording()
    r.assert_number(1 if rejection == "admission" else 2)
    assert workbench.panel.get_station_data(1) == before
    r.w.outcome.hardware_available, r.w.outcome.error = True, None
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 2)
    r.assert_number(2)


@pytest.mark.parametrize("advance_by", ["start", "analysis"])
def test_failed_store_blocks_uncommitted_number_and_retries_exact_transition(rounds, monkeypatch, advance_by):
    r = rounds
    first, second = r.record(), r.record()
    save = r.store.save

    def fail(snapshot):
        raise WorkbenchRoundStateError(f"Cannot save {r.store.path}")

    monkeypatch.setattr(r.store, "save", fail)
    if advance_by == "analysis":
        r.deliver(*first, *second)
    r.w.controller.start_recording()
    r.w.controller.refresh_recording_state()
    r.w.controller.refresh_recording_state()
    assert len(r.w.requests) == 2
    assert not r.w.states[-1][0]
    r.assert_number(1)
    assert len(r.errors) == 1
    assert not r.counter.can_record
    monkeypatch.setattr(r.store, "save", save)
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 2)
    r.assert_number(2)
    r.deliver(*first, *second)
    r.assert_number(2)


def test_bad_initial_state_is_diagnosed_once_and_can_be_retried(rounds):
    r = rounds
    r.store.path.write_text("broken", encoding="utf-8")
    r.w.controller.bind_round_counter(WorkbenchRoundCounter(r.store), r.shown.append, r.errors.append)
    for _ in range(3):
        r.w.controller.refresh_recording_state()
        r.w.controller.start_recording()
    assert r.w.requests == []
    assert len(r.errors) == 1
    assert r.store.path.read_text(encoding="utf-8") == "broken"
    r.store.save(RoundSnapshot(r.clock.value.date(), 9))
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 9)
    r.assert_number(9)


def test_completed_callback_can_start_next_round_after_empty_analysis_completion(rounds):
    r = rounds
    queues = {name: AnalysisTestQueue("wav", {"display_sequence": []})
              for name in r.w.config.analysis.test_queues}
    r.w.repository.result = configuration_with(r.w.config, test_queues=queues)
    r.w.controller.refresh_configuration()
    r.w.controller.start_recording()
    r.w.complete()
    completed = r.w.controller._recording.completed

    def finish_and_start(result):
        completed(result)
        r.w.controller.start_recording()

    r.w.controller._recording = replace(r.w.controller._recording, completed=finish_and_start)
    r.w.controller.start_recording()
    r.w.complete()
    assert len(r.w.requests) == 3
    assert_wav_number(r.w.requests[-1], 2)
    r.assert_number(2)


def test_source_refresh_abandons_partial_round_without_product_selector(rounds):
    r = rounds
    old = r.record()
    r.w.repository.source_identity = "another.json"
    r.w.controller.refresh_configuration()
    first, second = r.record(), r.record()
    r.deliver(*old, *first, *second)
    r.assert_number(2)


@pytest.mark.parametrize("terminal", ["failure", "cancelled"])
def test_first_condition_retry_after_midnight_uses_new_day(rounds, terminal):
    r = rounds
    first, second = r.record(), r.record()
    r.deliver(*first, *second)
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 2)
    r.w.complete(outcome=terminal)
    r.clock.value = datetime(2026, 11, 13)
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 1)
    r.assert_number(1)


def test_non_success_analysis_preserves_existing_diagnostic_and_does_not_advance(rounds):
    r = rounds
    first, second = r.record(), r.record()
    r.deliver(*second, *first[1:])
    r.service.events.append(result_event(first[0], failed=True))
    r.w.controller.poll_analysis_events()
    r.assert_number(1)
    assert r.events[-1]["payload"]["execution_status"] == "分析失败"


@pytest.mark.parametrize("change_config", [False, True])
def test_midnight_during_bridge_admission_keeps_prepared_identity(rounds, change_config):
    r = rounds
    first, second = r.record(), r.record()
    r.deliver(*first, *second)

    def midnight():
        r.w.outcome.on_start = None
        r.clock.value = datetime(2026, 11, 13)
        if change_config:
            r.w.repository.result = configuration_with(r.w.config, project_name="new product")
            r.w.controller.refresh_configuration()
        r.w.controller.refresh_recording_state()

    r.w.outcome.on_start = midnight
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 2)
    r.w.complete()
    r.w.controller.start_recording()
    assert_wav_number(r.w.requests[-1], 1 if change_config else 2)


def test_store_error_notifier_can_reenter_periodic_refresh_without_second_dialog(rounds):
    r = rounds
    r.store.path.write_text("broken", encoding="utf-8")
    notifications = []

    def notify(message):
        notifications.append(message)
        if len(notifications) == 1:
            r.w.controller.refresh_recording_state()

    r.w.controller.bind_round_counter(WorkbenchRoundCounter(r.store), r.shown.append, notify)
    assert len(notifications) == 1
    assert not r.w.states[-1][0]
    r.store.save(RoundSnapshot(r.clock.value.date(), 5))
    r.w.controller.refresh_recording_state()
    assert r.w.states[-1][0]
    r.store.path.write_text("broken again", encoding="utf-8")
    r.w.controller.bind_round_counter(WorkbenchRoundCounter(r.store), r.shown.append, notify)
    assert len(notifications) == 2


def test_failed_initial_save_publishes_no_uncommitted_number_and_can_retry(rounds, monkeypatch):
    r = rounds
    r.store.path.unlink()
    save = r.store.save
    shown = []

    def fail(snapshot):
        raise WorkbenchRoundStateError(f"Cannot initialize {r.store.path}")

    monkeypatch.setattr(r.store, "save", fail)
    r.w.controller.bind_round_counter(WorkbenchRoundCounter(r.store), shown.append, r.errors.append)
    r.w.controller.start_recording()
    assert shown == []
    assert r.w.requests == []
    assert not r.store.path.exists()
    monkeypatch.setattr(r.store, "save", save)
    r.w.controller.start_recording()
    assert shown[-1] == 1
    assert_wav_number(r.w.requests[-1], 1)
    assert r.store.load() == RoundSnapshot(r.clock.value.date(), 1)


def test_unexpected_counter_errors_propagate(rounds, monkeypatch):
    r = rounds

    def programming_error(day):
        raise ValueError("counter programming bug")

    monkeypatch.setattr(r.counter, "refresh", programming_error)
    with pytest.raises(ValueError, match="counter programming bug"):
        r.w.controller.refresh_recording_state()
    assert r.errors == []
