"""Rejected attempts end without statistics and never join a later retest."""

from dataclasses import replace

import pytest

from unit_test.ui.test_workbench_result_sync import (
    sync, config, workbench, recording_workbench, paired_recording_workbench,
    result_event, early_event, instance, labels, configuration_with,
)
from unit_test.ui.test_workbench_round_counter_integration import rounds
from unit_test.ui.test_workbench_plot_results import plotted, plot_setup, start


@pytest.mark.parametrize("first_result", ["OK", "NG", None])
def test_precheck_rejection_cancels_statistics_and_retest_starts_fresh(sync, monkeypatch, first_result):
    first = sync.record()
    old_id = sync.w.controller._current_round_id
    if first_result:
        sync.deliver(result_event(first["right"], [instance(judgement=first_result)]))
    with monkeypatch.context() as patch:
        patch.setattr("base.recording_calibration_snapshot.load_mic_channel_calibrations", lambda *args: {})
        sync.w.controller.start_recording()
        sync.w.complete()
    assert sync.events[-1]["payload"]["precheck_failed"]
    assert len(sync.w.completions) == 2  # Saved recordings remain registered.
    assert sync.w.controller._condition_index == 0
    assert sync.w.controller._current_round_id is None
    assert labels(sync.summary)[0] == "0"
    assert sync.panel.get_station_data(1)["result"] == "--"
    sync.w.complete()  # Duplicate archive/completion must not count or resume.
    for task in first.values():
        sync.deliver(early_event(task), result_event(task, [instance()]))
    assert not sync.w.controller._analysis_tasks
    assert labels(sync.summary)[0] == "0"
    newer = sync.record()
    assert sync.w.controller._current_round_id != old_id
    assert newer["right"]["condition_id"] == first["right"]["condition_id"]
    sync.deliver(result_event(newer["right"], [instance()]))
    assert labels(sync.summary)[0] == "0"
    last = sync.record()
    sync.deliver(result_event(last["right"], [instance()]))
    assert labels(sync.summary)[:4] == ("1", "1", "0", "0")


def test_rejection_after_last_recording_releases_round_counter(rounds, monkeypatch):
    first = rounds.record()
    old_id = rounds.w.controller._current_round_id
    with monkeypatch.context() as patch:
        patch.setattr("base.recording_calibration_snapshot.load_mic_channel_calibrations", lambda *args: {})
        rounds.w.controller.start_recording()
        rounds.w.complete()
    assert rounds.counter._rounds[old_id].abandoned
    rounds.assert_number(1)
    rounds.deliver(*first)
    rounds.assert_number(1)
    newer = rounds.record()
    assert newer[0]["condition_id"] == first[0]["condition_id"]
    assert rounds.w.controller._current_round_id != old_id


def test_old_rejection_does_not_cancel_new_round(sync):
    sync.record()
    old = sync.w.controller._recording_round
    sync.w.controller._cancel_rejected_round(old)
    sync.record()
    current = sync.w.controller._recording_round
    sync.w.controller._cancel_rejected_round(old)
    assert sync.w.controller._recording_round is current
    assert sync.w.controller._condition_index == 1


def test_completed_unaffected_station_keeps_its_record(sync, monkeypatch):
    conditions = sync.w.config.analysis.context.conditions
    sync.w.repository.result = configuration_with(sync.w.config, conditions=(
        replace(conditions[0], applicable_station_ids=("right",)),
        replace(conditions[1], applicable_station_ids=("left",)),
    ))
    sync.w.controller.refresh_configuration()
    first = sync.record()
    for task in first.values():
        sync.deliver(result_event(task, [instance()]))
    assert labels(sync.summary)[:4] == ("1", "1", "0", "0")
    monkeypatch.setattr("base.recording_calibration_snapshot.load_mic_channel_calibrations", lambda *args: {})
    sync.w.controller.start_recording()
    sync.w.complete()
    assert labels(sync.summary)[:4] == ("1", "1", "0", "0")
    assert sync.panel.get_station_data(1)["result"] == "OK"
    assert sync.panel.get_station_data(2)["result"] == "--"


def test_service_rejection_uses_same_no_statistics_cleanup(sync, monkeypatch):
    monkeypatch.setattr(sync.service, "submit_batch", lambda payload: {
        "accepted": False, "accepted_task_ids": [], "issues": [],
    })
    sync.w.controller.start_recording()
    sync.w.complete()
    assert sync.w.controller._condition_index == 0
    assert sync.w.controller._recording_round is None
    assert labels(sync.summary)[0] == "0"


def test_rejected_round_stops_pending_plot_states(plotted, monkeypatch):
    start(plotted)
    plotted.complete()
    monkeypatch.setattr("base.recording_calibration_snapshot.load_mic_channel_calibrations", lambda *args: {})
    start(plotted)
    plotted.complete()
    for cell in plotted.grid._cells.values():
        assert cell._condition_states["warm"][0] == "本轮已终止"
        assert cell._condition_states["cold"][0] == "检查未通过"
