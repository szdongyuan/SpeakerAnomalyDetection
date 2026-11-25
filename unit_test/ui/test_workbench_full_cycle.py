"""Full-cycle sequencing uses capture time, independently of archive completion."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from base.recording_process_protocol import RecordingCaptureTiming
from base.workbench_config_values import WorkbenchExecution
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_round_counter_integration import rounds


class Timer:
    def __init__(self):
        self.now = 100.0
        self.callback = None
        self.due = None

    def schedule(self, seconds, callback):
        self.due = self.now + seconds
        self.callback = callback

    def cancel(self):
        self.callback = None

    def fire(self, now):
        self.now = now
        callback, self.callback = self.callback, None
        assert callback is not None
        callback()


@pytest.fixture
def cycle(recording_workbench):
    return enable_cycle(recording_workbench)


def enable_cycle(w):
    w.config = configuration_with(w.config, execution=WorkbenchExecution("full_cycle", .5))
    w.repository.result = w.config
    w.controller.refresh_configuration()
    w.timer = Timer()
    w.controller.bind_execution_timer(w.timer.schedule, w.timer.cancel, clock=lambda: w.timer.now)
    w.cancellations = []
    w.service.cancel = w.cancellations.append

    def captured(index=0, start=100.0, end=102.0, *, release=True):
        request = w.requests[index]
        timing = RecordingCaptureTiming(request.request_id, 1, start, end, request.target_samples)
        session = SimpleNamespace(request=request)
        w.callbacks[index].capture_timing(session, timing)
        if release:
            w.callbacks[index].capture_available(session, timing)
        return timing

    w.captured = captured
    return w


def test_capture_end_drives_next_condition_before_save_and_archive(cycle):
    w = cycle
    w.controller.start_recording()
    w.captured()
    assert w.timer.due == 102.5
    w.timer.fire(102.49)
    assert len(w.requests) == 1
    w.timer.fire(102.5)
    assert [r.station_group.condition_id for r in w.requests] == ["warm", "cold"]
    assert not w.archive_submissions
    w.captured(1, 102.6, 104.6)
    w.complete(1)
    assert w.controller.automatic_execution_busy
    w.complete(0)
    assert not w.controller.recording_busy
    assert w.timer.callback is None
    assert w.controller._condition_index == 0


def test_duplicate_start_timing_and_release_do_not_skip_conditions(cycle):
    w = cycle
    w.controller.start_recording()
    for _ in range(3):
        w.captured()
        w.controller.start_recording()
    w.timer.fire(102.5)
    assert len(w.requests) == 2
    w.captured(0)
    assert w.controller._condition_index == 1
    assert w.timer.callback is None


def test_waits_for_capture_release_and_service_capacity(cycle):
    w = cycle
    w.controller.start_recording()
    w.captured(release=False)
    w.timer.fire(102.5)
    assert len(w.requests) == 1
    w.service.can_start_recording = False
    w.captured()
    w.timer.fire(102.6)
    assert len(w.requests) == 1
    w.service.can_start_recording = True
    w.timer.fire(102.7)
    assert len(w.requests) == 2


def test_missing_capture_completion_event_stops_auto_but_preserves_saved_wav(cycle):
    w = cycle
    w.controller.start_recording()
    w.complete()
    assert not w.controller.automatic_execution_busy
    assert len(w.archive_submissions) == len(w.completions) == 1
    assert len(w.cleanup_problems) == 1
    assert w.timer.callback is None
    assert w.controller._condition_index == 0


@pytest.mark.parametrize("reason", ["stop", "config", "logout", "device"])
def test_invalidated_wait_cannot_restart_from_old_callback(cycle, reason):
    w = cycle
    w.controller.start_recording()
    w.captured()
    stale = w.timer.callback
    if reason == "stop":
        w.controller.stop_automatic_execution()
    elif reason == "config":
        w.repository.result = configuration_with(w.config, project_name="new")
        w.controller.refresh_configuration()
    elif reason == "logout":
        w.device.login_ready = False
        w.timer.fire(102.5)
    else:
        w.device.snapshot = replace(w.device.snapshot, generation=2)
        w.timer.fire(102.5)
    w.timer.now = 200
    stale()
    assert len(w.requests) == 1
    assert not w.controller.automatic_execution_busy
    w.complete()
    assert not w.controller.recording_busy


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_failure_of_prior_finalizer_cancels_new_capture(cycle, outcome):
    w = cycle
    w.controller.start_recording()
    w.captured()
    w.timer.fire(102.5)
    w.complete(0, outcome=outcome)
    assert not w.controller.automatic_execution_busy
    assert w.cancellations == [w.requests[1].request_id]
    w.complete(1, outcome="cancelled")
    assert not w.controller.recording_busy


def test_second_admission_failure_resets_round(cycle):
    w = cycle
    w.controller.start_recording()
    w.captured()
    w.outcome.error = RuntimeError("device busy")
    w.timer.fire(102.5)
    assert not w.controller.automatic_execution_busy
    assert w.controller._condition_index == 0
    assert w.cleanup_problems


def test_identity_is_frozen_for_entire_cycle(cycle):
    w = cycle
    w.controller.start_recording()
    w.identity.model = "changed"
    w.identity.serial = "changed"
    w.captured()
    w.timer.fire(102.5)
    first, second = w.requests
    assert all(r.station_group.stations[0].filename.endswith("_M-02_007_SN-008.wav")
               for r in (first, second))


@pytest.mark.parametrize("count,gap", [(1, .5), (2, 0), (3, .5)])
def test_configured_condition_count_and_zero_gap(cycle, count, gap):
    w = cycle
    conditions = w.config.recording.context.conditions[:count]
    if count == 3:
        conditions += (replace(conditions[0], condition_id="third", condition_name="third"),)
    w.config = configuration_with(w.config, conditions=conditions,
                                  execution=WorkbenchExecution("full_cycle", gap))
    w.repository.result = w.config
    w.controller.refresh_configuration()
    w.controller.start_recording()
    for index in range(count):
        end = 102.0 + index * (2 + gap)
        w.captured(index, end - 2, end)
        w.complete(index)
        if index < count - 1:
            assert len(w.requests) == index + 1  # Even zero gap uses the event loop.
            w.timer.fire(end + gap)
    assert len(w.requests) == count
    assert not w.controller.recording_busy


def test_manual_mode_does_not_advance_on_capture_event(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    request = w.requests[0]
    timing = RecordingCaptureTiming(request.request_id, 1, 100, 102, request.target_samples)
    w.callbacks[0].capture_timing(SimpleNamespace(request=request), timing)
    w.callbacks[0].capture_available(SimpleNamespace(request=request), timing)
    assert w.controller._condition_index == 0
    assert w.controller.recording_busy
    w.complete()
    assert w.controller._condition_index == 1
    assert len(w.requests) == 1


def test_late_save_does_not_reset_newer_condition_display(cycle, monkeypatch):
    w = cycle
    starts = []
    monkeypatch.setattr(w.controller._plot_presenter, "recording_started",
                        lambda context, **kwargs: starts.append(context.request_id))
    w.controller.start_recording()
    w.callbacks[0].started(SimpleNamespace(request=w.requests[0]))
    w.captured()
    w.timer.fire(102.5)
    w.callbacks[1].started(SimpleNamespace(request=w.requests[1]))
    w.captured(1, 102.5, 104.5)
    w.complete(1)
    w.complete(0)
    assert starts == [r.request_id for r in w.requests]
    assert not w.controller._recording_started_ids


def test_counter_waits_for_both_archives_and_analysis_results(rounds):
    r = rounds
    w = enable_cycle(r.w)
    archives = []
    w.outcome.on_archive = lambda result, done: archives.append((result, done))
    w.controller.start_recording()
    w.captured()
    w.timer.fire(102.5)
    w.captured(1, 102.5, 104.5)
    w.complete(1)
    w.complete(0)
    r.assert_number(1)
    result, done = archives[0]
    done(result.request_id, None)
    r.deliver(*r.service.submissions[0]["tasks"])
    assert w.controller.automatic_execution_busy
    r.assert_number(1)
    result, done = archives[1]
    done(result.request_id, None)
    assert not w.controller.automatic_execution_busy
    r.assert_number(1)
    r.deliver(*r.service.submissions[1]["tasks"])
    r.assert_number(2)
    done(result.request_id, None)
    r.assert_number(2)


def test_late_analysis_rejection_stops_following_capture(rounds):
    r = rounds
    w = enable_cycle(r.w)
    w.controller.start_recording()
    w.captured()
    w.timer.fire(102.5)
    r.service.submit_batch = lambda payload: {
        "accepted": False, "accepted_task_ids": [], "issues": []}
    w.complete(0)
    assert not w.controller.automatic_execution_busy
    assert w.cancellations == [w.requests[1].request_id]
    w.complete(1, outcome="cancelled")
    r.assert_number(1)
    assert w.controller._recording_round is None


def test_device_change_stops_active_capture_without_waiting_for_completion(cycle):
    w = cycle
    w.controller.start_recording()
    w.device.snapshot = None
    w.controller.refresh_recording_state()
    assert not w.controller.automatic_execution_busy
    assert w.cancellations == [w.requests[0].request_id]
    w.complete(outcome="cancelled")
    assert not w.controller.recording_busy
