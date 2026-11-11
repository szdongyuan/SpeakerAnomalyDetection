"""Service acceptance using independent resident Windows spawn workers."""

import copy
import csv
import multiprocessing
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.analysis.process_manager import AnalysisProcessManager
from base.analysis.runtime_protocol import RunTask, StopWorker, WorkerEvent
from base.analysis.service import AnalysisService


def test_zero_recovery_window_is_rejected():
    with pytest.raises(ValueError, match="restart_window"):
        AnalysisService(restart_window=0)


def pump(service, predicate, timeout=35):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(service.poll_events())
        if predicate(events):
            return events
        time.sleep(0.01)
    raise AssertionError(f"Service timed out: {service.station_states}")


def ready(service, ids):
    assert service.sync_stations(ids)["accepted"]
    pump(service, lambda _: all(service.station_states[s]["state"] == "ready" for s in ids))


def finish(service):
    if service.closed:
        return ()
    service.shutdown()
    return pump(service, lambda events: any(e["event_type"] == "service_closed" for e in events))


def results(events):
    return [event["payload"] for event in events if event["event_type"] == "task_result"]


@pytest.fixture
def real_payload(batch_payload, tmp_path):
    payload = copy.deepcopy(batch_payload)
    for station in payload["stations"]:
        station["channels"] = station["channels"][:2]
        for channel in station["channels"]:
            channel.update(signal_type="sound_pressure", v2pa_factor=2.0)
    rate = 44100
    tone = np.sin(2 * np.pi * 1000 * np.arange(3 * rate) / rate)
    for number, task in enumerate(payload["tasks"]):
        amplitude = (number + 1) * 0.01
        sf.write(task["wav_path"], np.column_stack((amplitude * tone, 2 * amplitude * tone)),
                 rate, subtype="PCM_24")
        task["analysis_result_directory"] = str(tmp_path / "results")
        task["analysis_config"] = {
            "display_sequence": ["SPL"],
            "SPL": {"type": "SPL", "analysis_channels": [0, 1], "weighting": "Z"},
        }
    return payload


def test_four_stations_two_conditions_real_wavs(real_payload):
    service = AnalysisService()
    ids = [s["station_id"] for s in real_payload["stations"]]
    try:
        ready(service, ids)
        pids = {s: service.station_states[s]["pid"] for s in ids}
        assert len(set(pids.values())) == 4
        receipt = service.submit_batch(real_payload)
        assert receipt["accepted"] and len(receipt["accepted_task_ids"]) == 8
        events = pump(service, lambda events: len(results(events)) == 8)
        actual = {r["task_id"]: r for r in results(events)}
        paths = set()
        for number, task in enumerate(real_payload["tasks"]):
            result = actual[task["task_id"]]
            assert result["execution_status"] == "分析完成"
            assert result["station_id"] == task["station_id"]
            assert result["condition_id"] == task["condition_id"]
            for channel, item in enumerate(result["instance_results"]):
                expected = 20 * np.log10((number + 1) * 0.02 * (channel + 1) / np.sqrt(2) / 20e-6)
                assert item["metrics"]["overall_spl"] == pytest.approx(expected, abs=0.001)
                artifact, curve_artifact = item["artifacts"]
                assert curve_artifact["kind"] == "curve_csv"
                assert curve_artifact["status"] == "saved"
                with open(curve_artifact["path"], encoding="utf-8-sig", newline="") as stream:
                    assert len(list(csv.DictReader(stream))) == len(item["display_payload"]["x"])
                assert artifact["status"] == "saved"
                paths.add(artifact["path"])
                with open(artifact["path"], encoding="utf-8-sig", newline="") as stream:
                    row, = csv.DictReader(stream)
                assert len(row) == 4
                assert float(row["总体声压级dB"]) == pytest.approx(expected, abs=0.001)
                assert Path(artifact["path"]).parent.name == Path(task["wav_path"]).stem
            assert service.station_states[task["station_id"]]["pid"] == pids[task["station_id"]]
        assert len(paths) == 16
        assert len(list(Path(real_payload["tasks"][0]["analysis_result_directory"]).rglob("*.csv"))) == 32
        for station in ids:
            started = [e["task_id"] for e in events
                       if e["event_type"] == "task_started" and e["station_id"] == station]
            assert started == [t["task_id"] for t in real_payload["tasks"] if t["station_id"] == station]
        assert service.submit_batch(real_payload)["issues"][0]["error_type"] == "DuplicateTask"
    finally:
        finish(service)
    assert service.closed
    assert not set(pids.values()) & {p.pid for p in multiprocessing.active_children()}
    print(f"Multi-station WAV verification: pids={list(pids.values())}, "
          "tasks=8, WAV=3s/44100Hz/2ch/PCM24, SPL results=16, CSV=32, closed=True")


def controlled_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    while True:
        command = connection.recv()
        if isinstance(command, StopWorker):
            connection.send(WorkerEvent("stopped", station_id, generation))
            connection.close()
            return
        assert isinstance(command, RunTask)
        task = command.task
        connection.send(WorkerEvent("task_started", station_id, generation, task.task_id))
        parameters = task.instances[0].parameters if task.instances else {}
        barrier = parameters.get("barrier_dir")
        if barrier:
            root = Path(barrier)
            root.mkdir(exist_ok=True)
            (root / station_id).touch()
            deadline = time.monotonic() + 15
            while len(list(root.iterdir())) < parameters["participants"]:
                if time.monotonic() > deadline:
                    os._exit(31)
                time.sleep(0.01)
        if task.task_id.startswith("crash"):
            os._exit(17)
        if task.task_id.startswith("hang"):
            time.sleep(60)
        time.sleep(0.1)
        connection.send(WorkerEvent("task_result", station_id, generation, task.task_id, {
            "task_id": task.task_id, "station_id": station_id,
            "condition_id": task.condition_id, "execution_status": "分析完成",
            "instance_results": [], "error": None,
        }))


def controlled_service(**kwargs):
    return AnalysisService(manager_factory=partial(
        AnalysisProcessManager, worker_target=controlled_worker,
    ), **kwargs)


def test_atomic_admission_capacity_and_snapshots(batch_payload):
    service = controlled_service(waiting_capacity=2)
    ids = [s["station_id"] for s in batch_payload["stations"]]
    try:
        assert not service.submit_batch(batch_payload)["accepted"]
        ready(service, ids)
        invalid = copy.deepcopy(batch_payload)
        invalid["tasks"][-1]["analysis_config"]["SPL 1"]["analysis_channels"] = [99]
        assert not service.submit_batch(invalid)["accepted"]
        assert service.pending_task_count == 0
        oversized = copy.deepcopy(batch_payload)
        oversized["tasks"].append(dict(oversized["tasks"][0], task_id="overflow"))
        assert not service.submit_batch(oversized)["accepted"]
        assert service.pending_task_count == 0
        assert service.submit_batch(batch_payload)["accepted"]
        batch_payload["tasks"][0]["task_id"] = "mutated"
        events = pump(service, lambda events: len(results(events)) == 8)
        assert "mutated" not in {r["task_id"] for r in results(events)}
        assert service.pending_task_count == 0
    finally:
        finish(service)


def task_subset(payload, ids):
    value = copy.deepcopy(payload)
    value["tasks"] = [t for t in value["tasks"] if t["station_id"] in ids]
    return value


def test_workers_actually_overlap_not_just_have_distinct_pids(batch_payload, tmp_path):
    service = controlled_service()
    ids = [s["station_id"] for s in batch_payload["stations"]]
    batch_payload["tasks"] = batch_payload["tasks"][:4]
    for task in batch_payload["tasks"]:
        task["analysis_config"]["SPL 1"].update(barrier_dir=str(tmp_path / "barrier"), participants=4)
    try:
        ready(service, ids)
        assert service.submit_batch(batch_payload)["accepted"]
        events = pump(service, lambda events: len(results(events)) == 4)
        assert all(r["execution_status"] == "分析完成" for r in results(events))
        first_result = next(i for i, e in enumerate(events) if e["event_type"] == "task_result")
        assert sum(e["event_type"] == "task_started" for e in events[:first_result]) == 4
    finally:
        finish(service)


def test_station_diff_busy_drain_and_readd(batch_payload):
    service = controlled_service()
    original = [s["station_id"] for s in batch_payload["stations"]]
    try:
        ready(service, original)
        pids = {s: service.station_states[s]["pid"] for s in original}
        assert service.submit_batch(batch_payload)["accepted"]
        before = pump(service, lambda e: sum(x["event_type"] == "task_started" for x in e) == 4)
        receipt = service.sync_stations(["station-1", "station-2", "station-5"])
        assert receipt["removed_station_ids"] == ["station-3", "station-4"]
        assert receipt["added_station_ids"] == ["station-5"]
        removed_task = task_subset(batch_payload, ["station-3"])
        for task in removed_task["tasks"]:
            task["task_id"] += "-new"
        assert not service.submit_batch(removed_task)["accepted"]
        # Re-add during draining: retain old generation until its accepted work finishes.
        assert service.sync_stations(["station-1", "station-2", "station-3", "station-5"])["accepted"]
        events = pump(service, lambda e: len(results(before + e)) == 8
                      and "station-4" not in service.station_states
                      and service.station_states["station-3"]["state"] == "ready")
        assert len(results(before + events)) == 8
        assert all(r["execution_status"] == "分析完成" for r in results(before + events))
        assert service.station_states["station-3"]["pid"] != pids["station-3"]
        for s in ("station-1", "station-2"):
            assert service.station_states[s]["pid"] == pids[s]
        assert service.sync_stations([])["accepted"]
        pump(service, lambda _: not service.station_states)
        assert not service.closed
        ready(service, ["station-1"])
        assert service.station_states["station-1"]["pid"] != pids["station-1"]
    finally:
        finish(service)


@pytest.mark.parametrize("first", ["crash", "hang"])
def test_failure_recovers_for_next_task_without_replay(batch_payload, first):
    service = controlled_service(task_timeout=0.8, max_restarts=1)
    payload = task_subset(batch_payload, ["station-1", "station-2"])
    payload["tasks"][0]["task_id"] = first + "-once"
    try:
        ready(service, ["station-1", "station-2"])
        original = service.station_states["station-1"]["pid"]
        assert service.submit_batch(payload)["accepted"]
        events = pump(service, lambda e: len(results(e)) == 4)
        bad = [r for r in results(events) if r["task_id"] == first + "-once"]
        assert len(bad) == 1 and bad[0]["execution_status"] == "分析失败"
        assert bad[0]["error"]["type"] == ("TaskTimeout" if first == "hang" else "WorkerExit")
        assert len([r for r in results(events) if r["execution_status"] == "分析完成"]) == 3
        assert service.station_states["station-1"]["pid"] != original
    finally:
        finish(service)


def test_repeated_crashes_exhaust_budget_and_fail_waiting(batch_payload):
    service = controlled_service(max_restarts=1)
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"][0]["task_id"] = "crash-1"
    payload["tasks"][1]["task_id"] = "crash-2"
    payload["tasks"].append(dict(payload["tasks"][1], task_id="never-started"))
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        events = pump(service, lambda e: len(results(e)) == 3)
        third = next(r for r in results(events) if r["task_id"] == "never-started")
        assert third["error"]["type"] == "WorkerUnavailable"
        assert service.station_states["station-1"]["state"] == "failed"
        assert service.sync_stations(["station-1"])["accepted"]
        for _ in range(10):
            assert not service.poll_events()
        assert service.station_states["station-1"]["pid"] is None
    finally:
        finish(service)


@pytest.mark.parametrize("operation", ["remove", "shutdown"])
def test_drain_deadline_is_not_extended_and_terminal_is_unique(batch_payload, operation):
    service = controlled_service(drain_timeout=0.1, shutdown_timeout=0.1)
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"][0]["task_id"] = "hang-current"
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        before = pump(service, lambda e: any(x["event_type"] == "task_started" for x in e))
        started = time.monotonic()
        events = list(before)
        while time.monotonic() - started < 10:
            if operation == "remove":
                service.sync_stations([])
            else:
                service.shutdown()
            tick = time.monotonic()
            events.extend(service.poll_events(max_events=1))
            assert time.monotonic() - tick < 0.5
            if not service.station_states and len(results(events)) == 2:
                break
            time.sleep(0.01)
        assert time.monotonic() - started < 5
        assert len(results(events)) == len({r["task_id"] for r in results(events)}) == 2
        assert all(r["execution_status"] == "分析失败" for r in results(events))
        if operation == "shutdown":
            assert not service.sync_stations(["station-2"])["accepted"]
            assert not service.submit_batch(payload)["accepted"]
    finally:
        if not service.closed:
            finish(service)


def test_slow_consumer_backpressure_and_recent_id_bound(batch_payload):
    service = controlled_service(waiting_capacity=2, recent_task_capacity=2)
    payload = task_subset(batch_payload, ["station-1"])
    try:
        ready(service, ["station-1"])
        for batch_number in range(3):
            current = copy.deepcopy(payload)
            for task in current["tasks"]:
                task["task_id"] += f"-{batch_number}"
            assert service.submit_batch(current)["accepted"]
            events = []
            deadline = time.monotonic() + 10
            while len(results(events)) < 2 and time.monotonic() < deadline:
                events.extend(service.poll_events(max_events=1))
                assert service.pending_task_count <= 2
                assert len(service._events) < 16
                time.sleep(0.01)
            assert len(results(events)) == 2
            assert service.pending_task_count == 0 and len(service._recent) == 2
        assert service.submit_batch(current)["issues"][0]["error_type"] == "DuplicateTask"
    finally:
        finish(service)


def test_invalid_station_sync_does_not_change_live_set(batch_payload):
    service = controlled_service()
    try:
        ready(service, ["station-1"])
        pid = service.station_states["station-1"]["pid"]
        for invalid in (["station-1", "station-1"], [None], [" "], "station-1"):
            assert not service.sync_stations(invalid)["accepted"]
            assert service.station_states["station-1"]["pid"] == pid
    finally:
        finish(service)


def test_full_spectrogram_and_spl_return_during_graceful_shutdown(real_payload):
    service = AnalysisService(shutdown_timeout=30)
    payload = task_subset(real_payload, ["station-1", "station-2"])
    for task in payload["tasks"]:
        config = task["analysis_config"]
        config["display_sequence"].append("Spec")
        config["Spec"] = {"type": "Spec", "analysis_channels": [0],
                          "n_fft": 2048, "hop_length": 256}
    try:
        ready(service, ["station-1", "station-2"])
        assert service.submit_batch(payload)["accepted"]
        service.shutdown()
        assert not service.submit_batch(payload)["accepted"]
        events = finish(service)
        assert len(results(events)) == 4
        for result in results(events):
            assert result["execution_status"] == "分析完成"
            matrix = result["instance_results"][-1]["display_payload"]
            assert np.shape(matrix["values_db"]) == (1025, len(matrix["time_s"]))
            assert len(matrix["time_s"]) > 500
        assert events[-1]["event_type"] == "service_closed"
        assert service.poll_events() == ()
    finally:
        finish(service)


def no_ready_worker(connection, station_id, generation):
    time.sleep(60)


def test_startup_timeout_exhausts_without_fictitious_task_result():
    service = AnalysisService(startup_timeout=0.05, max_restarts=1,
                              manager_factory=partial(AnalysisProcessManager, worker_target=no_ready_worker))
    try:
        service.sync_stations(["station-1"])
        events = pump(service, lambda e: service.station_states["station-1"]["state"] == "failed"
                      and service._stations["station-1"].exhausted)
        assert results(events) == []
        starts = [e for e in events if e["event_type"] == "station_status"
                  and e["payload"]["state"] == "starting"]
        assert len(starts) == 2
        assert any(e["payload"].get("error", {}).get("type") == "StartupTimeout"
                   for e in events if e["payload"].get("error"))
    finally:
        finish(service)


def test_process_start_exception_is_reported_and_does_not_abort_other_station(monkeypatch):
    service = controlled_service(max_restarts=0)
    service.sync_stations(["station-1", "station-2"])
    manager = service._stations["station-1"].manager
    original = manager._context.Process

    def create(**kwargs):
        if kwargs["args"][1] == "station-1":
            raise OSError("synthetic spawn failure")
        return original(**kwargs)

    monkeypatch.setattr(manager._context, "Process", create)
    try:
        events = pump(service, lambda e: service._stations["station-1"].exhausted
                      and service.station_states["station-2"]["state"] == "ready")
        assert any(e["payload"].get("error", {}).get("type") == "OSError"
                   for e in events if e["payload"].get("error"))
    finally:
        finish(service)


def test_dispatch_failure_has_one_terminal_and_next_task_runs(batch_payload, monkeypatch):
    service = controlled_service()
    payload = task_subset(batch_payload, ["station-1"])
    try:
        ready(service, ["station-1"])
        manager = service._stations["station-1"].manager
        original = manager.submit

        def fail_first(task):
            monkeypatch.setattr(manager, "submit", original)
            raise OSError("synthetic dispatch failure")

        monkeypatch.setattr(manager, "submit", fail_first)
        assert service.submit_batch(payload)["accepted"]
        events = pump(service, lambda e: len(results(e)) == 2)
        assert results(events)[0]["error"]["stage"] == "dispatch"
        assert results(events)[1]["execution_status"] == "分析完成"
    finally:
        finish(service)


def test_received_result_wins_over_deadline(batch_payload, monkeypatch):
    service = controlled_service(task_timeout=100)
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"] = payload["tasks"][:1]
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        service.poll_events()
        manager = service._stations["station-1"].manager
        buffered = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            buffered.extend(manager.poll_events())
            if any(e["event_type"] == "task_result" for e in buffered):
                break
            time.sleep(0.01)
        assert any(e["event_type"] == "task_result" for e in buffered)
        original = manager.poll_events

        def drain_once(**kwargs):
            monkeypatch.setattr(manager, "poll_events", original)
            return tuple(buffered)

        monkeypatch.setattr(manager, "poll_events", drain_once)
        service._stations["station-1"].dispatched_at = time.monotonic() - 101
        events = service.poll_events()
        assert len(results(events)) == 1
        assert results(events)[0]["execution_status"] == "分析完成"
        assert manager.state == "ready"
    finally:
        finish(service)


def test_service_import_does_not_load_ui_or_algorithms():
    script = """
import sys
from base.analysis.service import AnalysisService
assert AnalysisService().station_states == {}
for prefix in ('ui', 'PyQt5', 'numpy', 'scipy', 'soundfile', 'matplotlib', 'base.analysis.executor'):
    assert not any(k == prefix or k.startswith(prefix + '.') for k in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_unreaped_resources_prevent_false_service_closed(batch_payload, monkeypatch):
    service = controlled_service(shutdown_timeout=0.01)
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"] = payload["tasks"][:1]
    payload["tasks"][0]["task_id"] = "hang-before-close"
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        pump(service, lambda e: any(x["event_type"] == "task_started" for x in e))
        manager = service._stations["station-1"].manager
        process = manager._process
        terminate = process.terminate
        abort = manager.abort
        with monkeypatch.context() as patch:
            patch.setattr(process, "terminate", lambda: None)
            patch.setattr(process, "kill", lambda: None)
            patch.setattr(manager, "abort", lambda error: abort(error, kill_after=0.01, cleanup_timeout=0.03))
            service.shutdown()
            events = pump(service, lambda e: any(
                x["payload"].get("error", {}).get("type") == "CleanupTimeout"
                for x in e if x["payload"].get("error")
            ))
            assert len(results(events)) == 1
            assert not service.closed and not manager.closed
            assert not any(e["event_type"] == "service_closed" for e in events)
        terminate()
    finally:
        finish(service)


def test_removed_worker_is_not_restarted_after_its_last_task_crashes(batch_payload):
    service = controlled_service()
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"] = payload["tasks"][:1]
    payload["tasks"][0]["task_id"] = "crash-last"
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        service.sync_stations([])
        events = pump(service, lambda e: not service.station_states)
        assert len(results(events)) == 1
        assert not any(e["event_type"] == "station_status" and e["payload"]["state"] == "starting"
                       for e in events)
    finally:
        finish(service)


def test_shutdown_does_not_restart_crashed_worker_with_waiting_tasks(batch_payload):
    service = controlled_service()
    payload = task_subset(batch_payload, ["station-1"])
    payload["tasks"][0]["task_id"] = "crash-shutdown"
    try:
        ready(service, ["station-1"])
        assert service.submit_batch(payload)["accepted"]
        events = finish(service)
        assert len(results(events)) == 2
        assert results(events)[1]["error"]["type"] == "WorkerUnavailable"
        assert not any(e["event_type"] == "station_status" and e["payload"]["state"] == "starting"
                       for e in events)
    finally:
        finish(service)
