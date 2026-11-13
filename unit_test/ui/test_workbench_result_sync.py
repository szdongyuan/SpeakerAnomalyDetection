"""Exercise paired result consumption through recording, submission and polling."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QLabel, QTableWidget

from base.audio_label_sync import AudioLabelJob
from base.workbench_analysis_config import AnalysisTestQueue
from ui.sequence.motor_information_summary_panel import MotorInformationSummaryPanel
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import AnalysisServiceStub
from unit_test.ui.test_workbench_result_state import instance


def result_event(task, instances=(), *, failed=False):
    return {
        "event_type": "task_result", "task_id": task["task_id"],
        "station_id": task["station_id"],
        "payload": {
            "task_id": task["task_id"], "station_id": task["station_id"],
            "condition_id": task["condition_id"],
            "execution_status": "分析失败" if failed else "分析完成",
            "instance_results": list(instances),
            "error": {"stage": "wav_read", "message": "missing WAV"} if failed else None,
        },
    }


@pytest.fixture
def sync(paired_recording_workbench, workbench, monkeypatch):
    from ui.sequence import motor_workbench_controller as module

    w = paired_recording_workbench
    clock = SimpleNamespace(now=lambda: datetime(2026, 11, 11, 12))
    monkeypatch.setattr(module, "datetime", clock)
    service, events = AnalysisServiceStub(), []
    w.controller.bind_analysis_service(service, events.append)
    w.controller.activate_analysis()
    summary = MotorInformationSummaryPanel()
    published = []

    def publish(records, day):
        published.append((records, day))
        summary.set_daily_records(records, day)

    w.controller.bind_result_summary(publish)

    def record():
        w.controller.start_recording()
        w.complete()
        return {task["station_id"]: task for task in service.submissions[-1]["tasks"]}

    def deliver(*incoming):
        service.events.extend(incoming)
        w.controller.poll_analysis_events()

    yield SimpleNamespace(w=w, panel=workbench.panel, summary=summary, service=service,
                          events=events, record=record, deliver=deliver, clock=clock,
                          published=published)
    summary.close()
    summary.deleteLater()


def labels(summary):
    return tuple(getattr(summary, name + "_label").text()
                 for name in ("total", "ok", "ng", "unmarked", "rate"))


@pytest.mark.parametrize("reverse_first", [False, True])
@pytest.mark.parametrize("forward,reverse,final_result,expected", [
    ("OK", "NG", "NG", ("1", "0", "1", "0", "0.0%")),
    ("OK", "--", "OK", ("1", "1", "0", "0", "100.0%")),
    ("NG", "--", "NG", ("1", "0", "1", "0", "0.0%")),
    ("--", "--", "--", ("1", "0", "0", "1", "0.0%")),
])
def test_station_updates_immediately_final_result_and_summary_wait_for_both(
        sync, reverse_first, forward, reverse, final_result, expected):
    first, second = sync.record(), sync.record()
    events = [result_event(first["right"], [instance(judgement=forward, metrics={"overall_spl": 0, "unit": "dB(A)"})]),
              result_event(second["right"], [instance(judgement=reverse, metrics={"overall_spl": 64.125})])]
    if reverse_first:
        events.reverse()
    sync.deliver(events[0])
    assert labels(sync.summary) == ("0", "0", "0", "0", "--")
    assert sync.published == []
    right = sync.panel.get_station_data(1)
    assert right["reverse" if reverse_first else "forward"] == (reverse if reverse_first else forward)
    assert right["rows"][0][2 if reverse_first else 1] == ("64.12" if reverse_first else "0.00 dB(A)")
    direction_label = "stationReverse1" if reverse_first else "stationForward1"
    assert sync.panel.findChild(QLabel, direction_label).text() == (reverse if reverse_first else forward)
    table = sync.panel.findChild(QTableWidget, "stationTable1")
    assert table.item(0, 2 if reverse_first else 1).text() == ("64.12" if reverse_first else "0.00 dB(A)")
    assert right["result"] == "--"
    assert sync.panel.findChild(QLabel, "stationResult1").text() == "--"
    sync.deliver(events[1])
    assert labels(sync.summary) == expected
    assert len(sync.published) == 1
    assert sync.panel.get_station_data(1)["rows"] == [["SPL", "0.00 dB(A)", "64.12"]]
    assert sync.panel.get_station_data(1)["result"] == final_result
    assert sync.panel.findChild(QLabel, "stationResult1").text() == final_result
    assert sync.panel.get_station_data(2)["forward"] == "--"
    sync.deliver(*events, result_event({**first["right"], "task_id": "unknown"}))
    assert labels(sync.summary) == expected and len(sync.published) == 1


def test_multiple_stations_and_rounds_accumulate_without_waiting_for_other_stations(sync):
    first, second = sync.record(), sync.record()
    sync.deliver(result_event(second["left"], [instance(judgement="NG")]),
                 result_event(first["right"], [instance()]))
    assert labels(sync.summary)[0] == "0"
    sync.deliver(result_event(first["left"], [instance()]))
    assert labels(sync.summary) == ("1", "0", "1", "0", "0.0%")
    left_before = sync.panel.get_station_data(2)
    sync.deliver(result_event(second["right"], [instance()]))
    assert sync.panel.get_station_data(2) == left_before
    assert labels(sync.summary) == ("2", "1", "1", "0", "50.0%")
    newer_first, newer_second = sync.record(), sync.record()
    sync.deliver(result_event(newer_first["right"], failed=True),
                 result_event(newer_second["right"], failed=True))
    assert labels(sync.summary) == ("3", "1", "1", "1", "33.3%")
    assert sync.panel.get_station_data(1)["rows"] == [["SPL", "--", "--"]]


def test_old_reverse_latest_task_cannot_overwrite_new_round_but_completes_history(sync):
    first, second = sync.record(), sync.record()
    sync.deliver(result_event(first["right"], [instance(metrics={"overall_spl": 10})]))
    newer_first = sync.record()
    sync.deliver(result_event(newer_first["right"], [instance(metrics={"overall_spl": 30})]))
    before = sync.panel.get_station_data(1)
    sync.deliver(result_event(second["right"], [instance(judgement="NG", metrics={"overall_spl": 20})]))
    assert sync.events[-1]["is_current"] is False
    assert sync.panel.get_station_data(1) == before
    assert before["rows"] == [["SPL", "30.00", "--"]]
    assert labels(sync.summary) == ("1", "0", "1", "0", "0.0%")
    newer_second = sync.record()
    sync.deliver(result_event(newer_second["right"], [instance()]))
    assert labels(sync.summary) == ("2", "1", "1", "0", "50.0%")


@pytest.mark.parametrize("location,field", [
    ("event", "station_id"), ("event", "condition_id"),
    ("payload", "station_id"), ("payload", "condition_id"), ("payload", "task_id"),
])
def test_mismatched_identity_is_diagnosed_before_context_merge(sync, caplog, location, field):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    first, second = sync.record(), sync.record()
    incoming = result_event(first["right"], [instance()])
    (incoming if location == "event" else incoming["payload"])[field] = "wrong-identity"
    before = sync.panel.get_station_data(1)
    sync.deliver(incoming)
    assert sync.panel.get_station_data(1) == before
    assert labels(sync.summary)[0] == "0"
    assert first["right"]["task_id"] in caplog.text
    assert field in caplog.text and "wrong-identity" in caplog.text
    assert "right" in caplog.text and "warm" in caplog.text
    assert jobs == []
    assert sync.events[-1]["payload"] is incoming["payload"]
    sync.deliver(result_event(second["right"], [instance()]))
    assert labels(sync.summary)[0] == "0"


def test_minimal_legacy_event_forwards_with_diagnostic_without_receiving_terminal(sync, caplog):
    first, second = sync.record(), sync.record()
    sync.deliver({"event_type": "task_result", "task_id": first["right"]["task_id"], "payload": {}})
    assert sync.events[-1]["is_current"] is True
    assert "station_id" in caplog.text and first["right"]["task_id"] in caplog.text
    sync.deliver(result_event(second["right"], [instance()]))
    assert labels(sync.summary)[0] == "0"


@pytest.mark.parametrize("change", ["configuration", "source", "same snapshot"])
def test_refresh_keeps_history_and_current_display_obeys_configuration_identity(sync, change):
    from ui.sequence.motor_workbench_controller import WorkbenchProductSelectionPorts

    w = sync.w
    w.controller.bind_product_selection(WorkbenchProductSelectionPorts(
        product_manager=None, show_catalog=lambda *_: None, set_enabled=lambda *_: None,
        is_busy=lambda: False, is_closing=lambda: False, notify_error=lambda *_: None))
    first, second = sync.record(), sync.record()
    sync.deliver(result_event(first["right"], [instance(metrics={"overall_spl": 10})]))
    if change == "configuration":
        w.repository.result = configuration_with(w.config, project_name="changed")
    elif change == "source":
        w.repository.source_identity = "another.json"
    w.identity.sample = "new identity"
    w.controller.refresh_configuration()
    before = sync.panel.get_station_data(1)
    sync.deliver(result_event(second["right"], [instance(judgement="NG")]))
    if change == "same snapshot":
        assert sync.events[-1]["is_current"] is True
        assert sync.panel.get_station_data(1)["result"] == "NG"
    else:
        assert sync.events[-1]["is_current"] is False
        assert sync.panel.get_station_data(1) == before
    assert labels(sync.summary)[0] == "1"
    assert sync.published[-1][0][0]["product_code"] == "007"


def test_cross_midnight_records_use_first_wav_date_and_publish_current_local_day(sync):
    sync.clock.now = lambda: datetime(2026, 11, 11, 23, 59, 59)
    first = sync.record()
    sync.clock.now = lambda: datetime(2026, 11, 12, 0, 0, 3)
    second = sync.record()
    assert Path(first["right"]["wav_path"]).name.startswith("2026-11-11_23-59-59")
    assert Path(second["right"]["wav_path"]).name.startswith("2026-11-12_00-00-03")
    assert [entry[2].isoformat() for entry in sync.w.archive_submissions] == ["2026-11-11", "2026-11-12"]
    sync.deliver(result_event(second["right"], [instance()]))
    assert not sync.published
    sync.deliver(result_event(first["right"], [instance()]))
    records = sync.w.controller._result_state.records_for(datetime(2026, 11, 11).date())
    assert len(records) == 1 and records[0]["completed_at"] == datetime(2026, 11, 11, 23, 59, 59)
    assert sync.w.controller._result_state.records_for(datetime(2026, 11, 12).date()) == []
    assert sync.summary.date_label.text() == "2026-11-12"
    assert labels(sync.summary) == ("0", "0", "0", "0", "--")


def test_failed_first_recording_retry_uses_retry_wav_timestamp(sync):
    sync.clock.now = lambda: datetime(2026, 11, 11, 23, 59, 59)
    sync.w.controller.start_recording()
    sync.w.complete(outcome="failure")
    sync.clock.now = lambda: datetime(2026, 11, 12, 0, 0, 3)
    first, second = sync.record(), sync.record()
    sync.deliver(result_event(second["right"], [instance()]), result_event(first["right"], [instance()]))
    assert labels(sync.summary)[0] == "1"
    record = sync.published[-1][0][0]
    assert record["completed_at"] == datetime(2026, 11, 12, 0, 0, 3)
    assert Path(first["right"]["wav_path"]).name.startswith("2026-11-12_00-00-03")


def test_export_failure_preserves_completed_values_and_outward_error(sync):
    first, second = sync.record(), sync.record()
    item = instance(metrics={"overall_spl": 42})
    item["artifacts"] = [{"status": "failed", "error": {"stage": "saving", "message": "access denied"}}]
    incoming = result_event(first["right"], [item])
    sync.deliver(incoming, result_event(second["right"], failed=True))
    assert sync.panel.get_station_data(1)["rows"] == [["SPL", "42.00", "--"]]
    assert sync.events[-2]["payload"] is incoming["payload"]
    assert labels(sync.summary) == ("1", "1", "0", "0", "100.0%")


def test_optional_summary_port_and_unexpected_consumer_errors(sync, monkeypatch):
    sync.w.controller.bind_result_summary(None)
    first, second = sync.record(), sync.record()
    sync.deliver(result_event(first["right"], [instance()]), result_event(second["right"], [instance()]))
    assert len(sync.w.controller._result_state.records_for(sync.clock.now().date())) == 1
    newer = sync.record()
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)

    def fail_render(*_):
        raise RuntimeError("unexpected rendering error")

    monkeypatch.setattr(sync.panel, "show_station_result", fail_render)
    with pytest.raises(RuntimeError, match="unexpected rendering error"):
        sync.deliver(result_event(newer["right"], [instance()]))
    assert jobs == [AudioLabelJob(newer["right"]["task_id"], newer["right"]["wav_path"], "OK")]
    # The first receive accepted this terminal even though rendering then failed.
    sync.deliver(result_event(newer["right"], [instance()]))
    assert len(jobs) == 1


@pytest.mark.parametrize("items,failed,label", [
    ([instance(judgement="OK"), instance("FFT", "FFT", "NG")], False, "NG"),
    ([instance(), instance("FFT", "FFT", None, execution_status="分析失败")], False, "OK"),
    ([instance(judgement="unknown")], False, "not_labeled"),
    ([], True, "not_labeled"),
    ([instance(judgement=None, execution_status="分析失败")], True, "not_labeled"),
    ([{**instance(), "artifacts": [{"status": "failed", "error": {
        "stage": "saving", "message": "access denied"}}]}], False, "OK"),
])
def test_one_direction_queues_its_own_wav_without_waiting(sync, items, failed, label):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    task = sync.record()["right"]
    incoming = result_event(task, items, failed=failed)
    sync.deliver(incoming)
    assert jobs == [AudioLabelJob(task["task_id"], task["wav_path"], label)]
    assert labels(sync.summary)[0] == "0"
    assert sync.events[-1]["payload"] is incoming["payload"]
    sync.deliver(incoming, result_event({**task, "task_id": "unknown"}, items))
    assert len(jobs) == 1


@pytest.mark.parametrize("payload", [None, [], {}, {"condition_id": "warm"}])
def test_invalid_result_has_no_label_job(sync, payload):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    task = sync.record()["right"]
    sync.deliver({**result_event(task), "payload": payload})
    assert jobs == []


@pytest.mark.parametrize("change", ["round", "configuration", "source"])
def test_late_result_uses_frozen_wav_after_current_identity_changes(sync, change):
    from ui.sequence.motor_workbench_controller import WorkbenchProductSelectionPorts

    w = sync.w
    w.controller.bind_product_selection(WorkbenchProductSelectionPorts(
        product_manager=None, show_catalog=lambda *_: None, set_enabled=lambda *_: None,
        is_busy=lambda: False, is_closing=lambda: False, notify_error=lambda *_: None))
    jobs = []
    w.controller.bind_audio_label_sync(jobs.append)
    first, second = sync.record(), sync.record()
    old_task = second["right"]
    old_path = old_task["wav_path"]
    # The backend's mutable submission data must not own the controller's path.
    old_task["wav_path"] = "changed-by-backend.wav"
    if change == "configuration":
        w.repository.result = configuration_with(w.config, project_name="new product")
    elif change == "source":
        w.repository.source_identity = "another.json"
    w.identity.sample = "new sample"
    w.controller.refresh_configuration()
    new_task = sync.record()["right"]
    assert new_task["wav_path"] != old_path
    sync.deliver(result_event(new_task, [instance()]))
    before = sync.panel.get_station_data(1)
    sync.deliver(result_event(old_task, [instance(judgement="NG")]))
    assert sync.events[-1]["is_current"] is False
    assert sync.panel.get_station_data(1) == before
    assert jobs == [AudioLabelJob(new_task["task_id"], new_task["wav_path"], "OK"),
                    AudioLabelJob(old_task["task_id"], old_path, "NG")]


def test_shutdown_still_queues_accepted_analysis_terminal(sync):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    task = sync.record()["right"]
    sync.w.controller.begin_analysis_shutdown()
    sync.deliver(result_event(task, [instance()]))
    assert jobs == [AudioLabelJob(task["task_id"], task["wav_path"], "OK")]


@pytest.mark.parametrize("outcome", ["rejected", "skipped"])
def test_rejected_or_skipped_analysis_has_no_label_job(sync, outcome):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    if outcome == "rejected":
        def reject(payload):
            sync.service.submissions.append(payload)
            return {"accepted": False, "accepted_task_ids": [], "issues": []}
        sync.service.submit_batch = reject
    else:
        queues = {name: AnalysisTestQueue("wav", {"display_sequence": []})
                  for name in sync.w.config.analysis.test_queues}
        sync.w.repository.result = configuration_with(sync.w.config, test_queues=queues)
        sync.w.controller.refresh_configuration()
    sync.w.controller.start_recording()
    sync.w.complete()
    assert not sync.w.controller._analysis_tasks
    if sync.service.submissions:
        sync.deliver(*(result_event(task, [instance()])
                       for task in sync.service.submissions[-1]["tasks"]))
    assert jobs == []


def test_optional_label_sink_can_be_unbound(sync):
    jobs = []
    sync.w.controller.bind_audio_label_sync(jobs.append)
    sync.w.controller.bind_audio_label_sync(None)
    task = sync.record()["right"]
    sync.deliver(result_event(task, [instance()]))
    assert sync.panel.get_station_data(1)["forward"] == "OK"
    assert jobs == []
