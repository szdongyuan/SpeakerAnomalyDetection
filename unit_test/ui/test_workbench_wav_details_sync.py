"""Saved-WAV details follow authenticated recording and analysis events."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from base.audio_label_sync import AudioLabelJob
from base.motor_daily_wav_records import DailyWavRecordStore
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from ui.sequence.motor_workbench_controller import MotorWorkbenchController, WorkbenchProductSelectionPorts
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_result_sync import sync, result_event, labels
from unit_test.ui.test_workbench_result_state import instance


@pytest.fixture
def wav_sync(sync):
    published = []

    def publish(records, day):
        published.append((records, day))
        sync.summary.set_daily_wav_records(records, day)

    sync.w.controller.bind_wav_details(publish)
    sync.wav_published = published
    return sync


def rows(sync, station=None):
    records = sync.wav_published[-1][0]
    return [row for row in records if station is None or row["station_id"] == station]


def test_saved_files_publish_before_results_with_independent_round_summary(wav_sync):
    s = wav_sync
    first_time = datetime(2026, 11, 11, 12, 0, 1)
    s.clock.now = lambda: first_time
    first = s.record()
    assert len(rows(s)) == 3
    assert rows(s, 1) == [dict(completed_at=first_time, station_id=1,
                             file_name=Path(first["right"]["wav_path"]).name,
                             condition="正转", result="未标记")]
    assert labels(s.summary) == ("0", "0", "0", "0", "--")
    assert s.published == []
    s.deliver(result_event(first["right"], [instance()]))
    assert rows(s, 1)[0]["result"] == "OK"
    assert rows(s, 2)[0]["result"] == "未标记"
    assert s.published == []

    second_time = datetime(2026, 11, 11, 12, 0, 5)
    s.clock.now = lambda: second_time
    second = s.record()
    assert len(rows(s)) == 6
    assert rows(s, 1)[1] == dict(completed_at=second_time, station_id=1,
                               file_name=Path(second["right"]["wav_path"]).name,
                               condition="反转", result="未标记")
    s.deliver(result_event(second["right"], [instance(judgement="NG")]))
    assert labels(s.summary) == ("1", "0", "1", "0", "0.0%")
    wav_store = DailyWavRecordStore()
    wav_store.replace(rows(s, 1), second_time.date())
    assert (wav_store.summary.total, wav_store.summary.ok, wav_store.summary.ng,
            wav_store.summary.pass_rate) == (2, 1, 1, "50.0%")


@pytest.mark.parametrize("asynchronous,error_kind", [
    (False, None), (True, None), (False, "callback"), (True, "callback"), (False, "submit"),
])
def test_files_register_before_archive_and_survive_archive_errors(wav_sync, asynchronous, error_kind):
    s = wav_sync
    pending = []

    def archive(completion, callback):
        assert len(rows(s)) == len(completion.stations)
        assert all(row["result"] == "未标记" for row in rows(s))
        assert not s.service.submissions
        if error_kind == "submit":
            raise OSError("archive unavailable")
        if asynchronous:
            pending.append((completion, callback))
        else:
            callback(completion.request_id, OSError("archive unavailable") if error_kind else None)

    s.w.outcome.on_archive = archive
    s.w.controller.start_recording()
    s.w.complete()
    publication_count = len(s.wav_published)
    s.w.complete()  # Duplicate completion before or after the archive callback.
    assert len(s.wav_published) == publication_count
    assert len(rows(s)) == 3
    if asynchronous:
        assert s.w.controller.recording_busy and not s.service.submissions
        completion, callback = pending.pop()
        callback(completion.request_id, OSError("archive unavailable") if error_kind else None)
        callback(completion.request_id, None)
    assert not s.w.controller.recording_busy
    assert len(s.w.completions) == 1 and len(s.service.submissions) == 1
    assert bool(s.w.archive_problems) == bool(error_kind)
    assert len(rows(s)) == 3 and s.published == []


def test_synchronous_capture_archive_and_first_poll_cannot_lose_results(wav_sync):
    s = wav_sync
    submit = s.service.submit_batch

    def submit_and_finish(payload):
        assert len(rows(s)) == len(payload["tasks"])
        assert all(row["result"] == "未标记" for row in rows(s))
        s.service.events.extend(result_event(task, [instance()]) for task in payload["tasks"])
        return submit(payload)

    s.service.submit_batch = submit_and_finish
    s.w.outcome.on_start = lambda: s.w.complete()
    s.w.controller.bind_analysis_input(lambda _: s.w.controller.poll_analysis_events())
    s.w.controller.start_recording()
    assert len(rows(s)) == 3 and all(row["result"] == "OK" for row in rows(s))
    assert s.published == [] and not s.w.controller.recording_busy


@pytest.mark.parametrize("mode", ["no items", "rejected", "not submitted"])
def test_skipped_or_unsubmitted_analysis_keeps_saved_unmarked_files(wav_sync, monkeypatch, mode):
    s = wav_sync
    if mode == "no items":
        queues = {name: AnalysisTestQueue("wav", {"display_sequence": []})
                  for name in s.w.config.analysis.test_queues}
        s.w.repository.result = configuration_with(s.w.config, test_queues=queues)
        s.w.controller.refresh_configuration()
    elif mode == "rejected":
        s.service.submit_batch = lambda _: {"accepted": False, "accepted_task_ids": [], "issues": []}
    else:
        from base.workbench_analysis_input import WorkbenchAnalysisInputError
        from ui.sequence import motor_workbench_controller as module

        def reject_input(*_, **__):
            raise WorkbenchAnalysisInputError("invalid saved analysis input")

        monkeypatch.setattr(module, "build_analysis_payload", reject_input)
    s.w.controller.start_recording()
    s.w.complete()
    assert len(rows(s)) == 3 and all(row["result"] == "未标记" for row in rows(s))
    assert not s.w.controller._analysis_tasks and not s.published


def test_no_analysis_service_or_consumer_still_retains_saved_files(recording_workbench):
    w = recording_workbench
    w.controller.start_recording()
    w.complete()
    published = []
    w.controller.bind_wav_details(lambda records, day: published.append((records, day)))
    saved = published[-1][0]
    assert len(saved) == len(w.completions[-1].stations)
    assert all(row["result"] == "未标记" for row in saved)
    w.controller.bind_wav_details(None)
    w.controller.start_recording()
    w.complete()
    w.controller.bind_wav_details(lambda records, day: published.append((records, day)))
    assert len(published[-1][0]) == len(saved) + len(w.completions[-1].stations)


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_failed_or_cancelled_capture_does_not_invent_files(wav_sync, outcome):
    s = wav_sync
    before = len(s.wav_published)
    s.w.controller.start_recording()
    s.w.complete(outcome=outcome)
    assert len(s.wav_published) == before and rows(s) == []
    assert not s.w.archive_submissions and not s.service.submissions
    s.record()
    assert len(rows(s)) == 3


def test_only_authenticated_completion_paths_register_with_frozen_mapping(wav_sync):
    s = wav_sync
    s.w.controller.start_recording()
    request = s.w.requests[-1]
    before = len(s.wav_published)
    s.w.complete(payload_id="unknown-request")
    assert rows(s) == [] and len(s.wav_published) == before
    # A committed configuration may change during capture. Use the admitted map.
    s.w.repository.result = configuration_with(
        s.w.config, stations=tuple(reversed(s.w.config.recording.context.stations)),
        conditions=tuple(reversed(s.w.config.recording.context.conditions)),
    )
    s.w.controller.refresh_configuration()
    group = request.station_group
    station = group.stations[0]
    actual = StationWav(station.station_id, str(Path(group.final_path) / "actual.saved.wav"), station.channels)
    completion = WorkbenchRecordingCompletion(
        request.request_id, group.project_id, group.condition_id, request.sample_rate, 8, (actual,),
    )
    callbacks = s.w.callbacks[-1]
    callbacks.workbench_completed(SimpleNamespace(request=request), completion)
    assert rows(s) == [dict(completed_at=datetime(2026, 11, 11, 12), station_id=1,
                           file_name="actual.saved.wav", condition="正转", result="未标记")]
    callbacks.workbench_completed(SimpleNamespace(request=request), completion)
    assert len(rows(s)) == 1


def test_identical_basenames_from_separate_requests_are_distinct(wav_sync):
    s = wav_sync
    for _ in range(2):
        s.w.controller.start_recording()
        request = s.w.requests[-1]
        group = request.station_group
        station = group.stations[0]
        actual = StationWav(station.station_id, str(Path(group.final_path) / request.request_id / "same.wav"),
                            station.channels)
        s.w.callbacks[-1].workbench_completed(SimpleNamespace(request=request), WorkbenchRecordingCompletion(
            request.request_id, group.project_id, group.condition_id, request.sample_rate, 8, (actual,),
        ))
    assert [row["file_name"] for row in rows(s)] == ["same.wav", "same.wav"]
    assert [row["condition"] for row in rows(s)] == ["正转", "反转"]


@pytest.mark.parametrize("items,failed,expected", [
    ([instance(), instance("FFT", "FFT", "NG")], False, "NG"),
    ([instance(), instance("FFT", "FFT", None, execution_status="分析失败")], True, "OK"),
    ([instance(judgement=None)], False, "未标记"),
    ([instance(judgement="unknown")], False, "未标记"),
    ([], True, "未标记"),
    ([{**instance(), "artifacts": [{"status": "failed", "error": {
        "stage": "saving", "message": "CSV access denied"}}]}], True, "OK"),
])
def test_single_wav_uses_shared_judgements_even_for_partial_and_csv_failures(wav_sync, items, failed, expected):
    s = wav_sync
    task = s.record()["right"]
    incoming = result_event(task, items, failed=failed)
    s.deliver(incoming)
    assert rows(s, 1)[0]["result"] == expected
    assert all(row["result"] == "未标记" for row in rows(s) if row["station_id"] != 1)
    publications = len(s.wav_published)
    s.deliver(incoming, result_event({**task, "task_id": "unknown"}, [instance(judgement="NG")]))
    assert len(s.wav_published) == publications
    assert rows(s, 1)[0]["result"] == expected and s.published == []


@pytest.mark.parametrize("location,field", [
    ("event", "station_id"), ("event", "condition_id"),
    ("payload", "station_id"), ("payload", "condition_id"), ("payload", "task_id"),
])
def test_mismatched_terminal_cannot_update_or_create_wav(wav_sync, location, field):
    s = wav_sync
    task = s.record()["right"]
    incoming = result_event(task, [instance()])
    (incoming if location == "event" else incoming["payload"])[field] = "incorrect"
    before = len(s.wav_published)
    s.deliver(incoming)
    assert len(s.wav_published) == before
    assert len(rows(s)) == 3 and all(row["result"] == "未标记" for row in rows(s))


@pytest.mark.parametrize("change", ["round", "configuration", "source"])
def test_late_result_updates_old_file_without_current_display_changes(wav_sync, change):
    s = wav_sync
    s.w.controller.bind_product_selection(WorkbenchProductSelectionPorts(
        product_manager=None, show_catalog=lambda *_: None, set_enabled=lambda *_: None,
        is_busy=lambda: False, is_closing=lambda: False, notify_error=lambda *_: None,
    ))
    first, second = s.record(), s.record()
    old = second["right"]
    frozen_name = Path(old["wav_path"]).name
    if change == "configuration":
        s.w.repository.result = configuration_with(s.w.config, project_name="new product")
    elif change == "source":
        s.w.repository.source_identity = "another.json"
    s.w.controller.refresh_configuration()
    s.clock.now = lambda: datetime(2026, 11, 11, 12, 0, 5)
    new = s.record()["right"]
    s.deliver(result_event(new, [instance()]))
    before = s.panel.get_station_data(1)
    s.deliver(result_event(old, [instance(judgement="NG")]))
    assert s.events[-1]["is_current"] is False
    assert s.panel.get_station_data(1) == before
    assert len(rows(s)) == 9
    assert next(row for row in rows(s, 1) if row["file_name"] == frozen_name)["result"] == "NG"
    assert next(row for row in rows(s, 1) if row["file_name"] == Path(new["wav_path"]).name)["result"] == "OK"
    s.deliver(result_event(first["right"], [instance()]))
    assert labels(s.summary) == ("1", "0", "1", "0", "0.0%")


def test_each_wav_keeps_its_own_date_and_late_results_publish_only_today(wav_sync):
    s = wav_sync
    first_time, second_time = datetime(2026, 11, 11, 23, 59, 59), datetime(2026, 11, 12, 0, 0, 3)
    s.clock.now = lambda: first_time
    first = s.record()
    s.clock.now = lambda: second_time
    second = s.record()
    assert len(rows(s)) == 3
    assert all(row["completed_at"] == second_time and row["condition"] == "反转" for row in rows(s))
    s.deliver(result_event(second["right"], [instance(judgement="NG")]))
    today_before = rows(s)
    s.deliver(result_event(first["right"], [instance()]))
    assert s.wav_published[-1][1] == second_time.date() and rows(s) == today_before
    yesterday = s.w.controller._wav_state.records_for(first_time.date())
    assert len(yesterday) == 3 and all(row["completed_at"] == first_time for row in yesterday)
    assert next(row for row in yesterday if row["station_id"] == 1)["result"] == "OK"
    round_records = s.w.controller._result_state.records_for(first_time.date())
    assert len(round_records) == 1 and round_records[0]["completed_at"] == first_time
    assert s.w.controller._result_state.records_for(second_time.date()) == []
    assert labels(s.summary) == ("0", "0", "0", "0", "--")


def test_controllers_own_independent_wav_ledgers(wav_sync):
    s = wav_sync
    s.record()
    other = MotorWorkbenchController(s.w.repository, s.panel, s.w.controller._plot_grid, lambda _: None, lambda: None)
    published = []
    other.bind_wav_details(lambda records, day: published.append((records, day)))
    assert published[-1][0] == []
    other.bind_recording(s.w.controller._recording)
    other.refresh_configuration()
    other.start_recording()
    s.w.complete()
    assert len(published[-1][0]) == 3
    assert len(rows(s)) == 3
    assert other._wav_state is not s.w.controller._wav_state


def test_unexpected_wav_consumer_failure_is_not_misreported_as_archive_failure(wav_sync):
    s = wav_sync

    def publish(records, _day):
        if records:
            raise RuntimeError("WAV render failed")

    s.w.controller.bind_wav_details(publish)
    s.w.controller.start_recording()
    with pytest.raises(RuntimeError, match="WAV render failed"):
        s.w.complete()
    assert not s.w.archive_problems and not s.w.archive_submissions
    assert len(s.w.controller._wav_state.records_for(s.clock.now().date())) == 3


@pytest.mark.parametrize("complete_pair", [False, True])
def test_wav_result_consumer_error_preserves_original_updates_and_first_acceptance(wav_sync, complete_pair):
    s = wav_sync
    task = s.record()["right"]
    if complete_pair:
        first, task = task, s.record()["right"]
        s.deliver(result_event(first, [instance()]))
    jobs, failed_publications = [], []
    s.w.controller.bind_audio_label_sync(jobs.append)
    judgement = "NG" if complete_pair else "OK"

    def publish(records, _day):
        if any(row["result"] == judgement for row in records):
            failed_publications.append(records)
            raise RuntimeError("WAV result render failed")

    s.w.controller.bind_wav_details(publish)
    incoming = result_event(task, [instance(judgement=judgement)])
    with pytest.raises(RuntimeError, match="WAV result render failed"):
        s.deliver(incoming)
    assert jobs == [AudioLabelJob(task["task_id"], task["wav_path"], judgement)]
    target = next(row for row in failed_publications[0] if row["file_name"] == Path(task["wav_path"]).name)
    assert target["result"] == judgement
    assert labels(s.summary) == (("1", "0", "1", "0", "0.0%") if complete_pair
                                 else ("0", "0", "0", "0", "--"))
    station = s.panel.get_station_data(1)
    assert station["forward"] == "OK"
    assert station["reverse"] == ("NG" if complete_pair else "--")
    assert station["result"] == ("NG" if complete_pair else "--")
    s.deliver(incoming)
    assert len(failed_publications) == 1 and len(jobs) == 1


@pytest.mark.parametrize("condition_count", [3, 4])
def test_all_configured_conditions_archive_analyze_and_publish_saved_files(wav_sync, condition_count):
    s = wav_sync
    first = s.w.config.recording.context.conditions[0]
    conditions = (*s.w.config.recording.context.conditions, *(
        replace(first, condition_id=f"extra-{index}", condition_name=f"负载升速 {index}")
        for index in range(3, condition_count + 1)
    ))
    s.w.repository.result = configuration_with(s.w.config, conditions=conditions)
    s.w.controller.refresh_configuration()
    batches = [s.record() for _ in conditions]
    expected = ["正转", "反转", *(condition.condition_name for condition in conditions[2:])]
    for station in (1, 2, 3):
        assert [row["condition"] for row in rows(s, station)] == expected
    assert len(s.w.archive_submissions) == len(s.w.completions) == len(batches) == condition_count
    assert len(s.service.submissions) == condition_count
    assert [completion.condition_id for completion in s.w.completions] == [c.condition_id for c in conditions]
    assert not s.w.controller.recording_busy
    s.deliver(*(result_event(task, [instance()]) for batch in batches for task in batch.values()))
    assert len(rows(s)) == condition_count * 3
    assert all(row["result"] == "OK" for row in rows(s))
    assert not s.w.controller._analysis_tasks


def test_inflight_third_condition_keeps_its_frozen_configured_name(wav_sync):
    s = wav_sync
    extra = replace(s.w.config.recording.context.conditions[0], condition_id="extra", condition_name="负载升速")
    conditions = (*s.w.config.recording.context.conditions, extra)
    s.w.repository.result = configuration_with(s.w.config, conditions=conditions)
    s.w.controller.refresh_configuration()
    s.record()
    s.record()
    s.w.controller.start_recording()
    assert s.w.requests[-1].station_group.condition_id == "extra"
    changed = (*conditions[:2], replace(extra, condition_name="已修改的工况名称"))
    s.w.repository.result = configuration_with(s.w.config, conditions=changed)
    s.w.controller.refresh_configuration()
    s.w.complete()
    assert [row["condition"] for row in rows(s, 1)] == ["正转", "反转", "负载升速"]
    assert len(s.w.archive_submissions) == len(s.w.completions) == len(s.service.submissions) == 3
    assert not s.w.controller.recording_busy
    third_tasks = s.service.submissions[-1]["tasks"]
    s.deliver(*(result_event(task, [instance(judgement="NG")]) for task in third_tasks))
    assert all(row["result"] == "NG" for row in rows(s) if row["condition"] == "负载升速")
