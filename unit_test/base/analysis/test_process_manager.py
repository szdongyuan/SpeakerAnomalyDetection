"""Spawned-process acceptance, using real WAVs and bounded lifecycle waits."""

import csv
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.process_manager import AnalysisProcessManager
from base.analysis.runtime_protocol import WorkerEvent, runtime_error


def wait_for(manager, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        events.extend(manager.poll_events())
        if predicate(events):
            return events
        time.sleep(0.01)
    raise AssertionError(f"Timed out: state={manager.state}, events={events}")


def wait_ready(manager):
    return wait_for(manager, lambda events: manager.state == "ready")


def wait_result(manager, task_id):
    events = wait_for(manager, lambda events: any(
        e["event_type"] == "task_result" and e["task_id"] == task_id for e in events
    ))
    results = [e["payload"] for e in events if e["event_type"] == "task_result"]
    assert len(results) == 1
    return results[0], events


@pytest.fixture
def task(tmp_path):
    rate = 44100
    t = np.arange(3 * rate) / rate
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
    wav = tmp_path / "three_seconds.wav"
    sf.write(wav, np.column_stack((tone, 2 * tone)), rate, subtype="PCM_24")
    return AnalysisTask(
        "forward-1", "station-1", "工位1", "forward", "正转", str(wav),
        str(tmp_path / "results"),
        tuple(AnalysisChannel(i, f"CH{i + 1}", "sound_pressure", v2pa_factor=2.0)
              for i in range(2)),
        tuple(AnalysisInstance("SPL", "SPL", i, {"weighting": "Z"}) for i in range(2)),
        {"mode": "none"},
    )


def test_real_wavs_reuse_pid_keep_order_and_save_per_channel(task):
    manager = AnalysisProcessManager(task.station_id)
    samples, rate = sf.read(task.wav_path, always_2d=True)
    scales = (1.0, 0.5, 1.5)
    tasks = []
    for number, (task_id, condition) in enumerate((
        ("forward-1", "forward"), ("reverse-1", "reverse"), ("forward-2", "forward"),
    )):
        path = Path(task.wav_path).with_name(f"{task_id}.wav")
        sf.write(path, samples * scales[number], rate, subtype="PCM_24")
        tasks.append(replace(task, task_id=task_id, condition_id=condition, wav_path=str(path)))
    durations = []
    try:
        startup = time.monotonic()
        pid = manager.start()
        assert manager.start() == pid
        wait_ready(manager)
        startup = time.monotonic() - startup
        observed = []
        for number, current in enumerate(tasks):
            started = time.monotonic()
            manager.submit(current)
            result, events = wait_result(manager, current.task_id)
            durations.append(round(time.monotonic() - started, 4))
            assert any(e["event_type"] == "task_started" for e in events)
            observed.append(result["task_id"])
            assert result["condition_id"] == current.condition_id
            assert result["execution_status"] == "分析完成"
            assert manager.pid == pid and manager.state == "ready"
            for index, item in enumerate(result["instance_results"]):
                expected = 20 * np.log10(0.2 * scales[number] * (index + 1) / np.sqrt(2) / 20e-6)
                assert item["metrics"]["overall_spl"] == pytest.approx(expected, abs=1e-4)
                assert item["metrics"]["wav_duration_sec"] == 3.0
                artifact, curve_artifact = item["artifacts"]
                assert curve_artifact["kind"] == "curve_csv"
                assert curve_artifact["status"] == "saved"
                with open(curve_artifact["path"], encoding="utf-8-sig", newline="") as stream:
                    assert len(list(csv.DictReader(stream))) == len(item["display_payload"]["x"])
                assert artifact["status"] == "saved"
                with open(artifact["path"], encoding="utf-8-sig", newline="") as stream:
                    row, = csv.DictReader(stream)
                assert len(row) == 4
                assert float(row["总体声压级dB"]) == pytest.approx(expected, abs=1e-4)
                assert f"_CH{index + 1}_" in Path(artifact["path"]).name
        assert observed == [t.task_id for t in tasks]
        assert len(list(Path(task.analysis_result_directory).rglob("*.csv"))) == 12
    finally:
        closing = manager.close()
    assert manager.closed and manager.pid is None
    assert any(e["event_type"] == "station_status" and e["payload"]["state"] == "closed"
               for e in closing)
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    assert f"analysis-io-{pid}" not in {thread.name for thread in threading.enumerate()}
    print(f"Resident WAV verification: pid={pid}, WAV=3x3s/44100Hz/2ch/PCM24, "
          f"startup_s={startup:.4f}, task_round_trip_s={durations}, csv_count=12, closed=True")


def test_broken_wav_does_not_kill_worker_and_snapshot_is_owned(task):
    manager = AnalysisProcessManager(task.station_id)
    try:
        pid = manager.start()
        wait_ready(manager)
        missing = replace(task, wav_path=task.wav_path + ".missing")
        manager.submit(missing)
        result, _ = wait_result(manager, missing.task_id)
        assert result["error"]["stage"] == "wav_read"
        assert manager.pid == pid
        next_task = replace(task, task_id="after-failure")
        manager.submit(next_task)
        next_task.instances[0].parameters["weighting"] = "invalid-after-submit"
        result, _ = wait_result(manager, next_task.task_id)
        assert result["execution_status"] == "分析完成"
        assert result["instance_results"][0]["metrics"]["weighting"] == "Z"
        assert manager.pid == pid
    finally:
        manager.close()


def test_one_inflight_task_and_wrong_station_are_rejected(task):
    manager = AnalysisProcessManager(task.station_id)
    try:
        with pytest.raises(RuntimeError):
            manager.submit(task)
        manager.start()
        wait_ready(manager)
        with pytest.raises(ValueError, match="station"):
            manager.submit(replace(task, station_id="other"))
        manager.submit(task)
        with pytest.raises(RuntimeError):
            manager.submit(replace(task, task_id="second"))
        result, _ = wait_result(manager, task.task_id)
        assert result["execution_status"] == "分析完成"
    finally:
        manager.close()


def test_close_while_busy_drains_full_spectrogram(task):
    manager = AnalysisProcessManager(task.station_id)
    try:
        manager.start()
        wait_ready(manager)
        spec = AnalysisInstance("Spec", "Spec", 0, {"n_fft": 2048, "hop_length": 256})
        manager.submit(replace(task, instances=(spec,)))
        events = manager.close(timeout=30)
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["execution_status"] == "分析完成"
        payload = result["instance_results"][0]["display_payload"]
        assert np.shape(payload["values_db"]) == (1025, len(payload["time_s"]))
        assert len(payload["time_s"]) > 500
        assert manager.closed
    finally:
        manager.close(timeout=0.2)


def crash_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    connection.recv()
    os._exit(17)


def hang_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    while True:
        time.sleep(1)


def exit_before_ready(connection, station_id, generation):
    connection.close()


def malformed_result_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    connection.recv()
    connection.send_bytes(b"not a pickle")
    connection.close()


@pytest.mark.parametrize("prior_failure", [False, True])
def test_transport_error_survives_worker_exit_without_overwriting_prior_failure(task, prior_failure):
    manager = AnalysisProcessManager(task.station_id, worker_target=malformed_result_worker)
    try:
        manager.start()
        wait_ready(manager)
        manager.submit(task)
        # Force the exited-worker path: IPC decoding and child exit finish before polling.
        manager._process.join(timeout=10)
        manager._thread.join(timeout=10)
        assert not manager._process.is_alive()
        assert not manager._thread.is_alive()
        assert manager._process.exitcode == 0
        original_error = runtime_error("execution", "TaskTimeout", "task deadline exceeded")
        if prior_failure:
            manager._fail(original_error)

        events = wait_for(manager, lambda events: manager.closed)
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["execution_status"] == "分析失败"
        error = result["error"]
        if prior_failure:
            assert error == original_error
        else:
            assert error["stage"] == "communication"
            assert error["type"] == "UnpicklingError"
            assert "invalid load key" in error["message"]
        closed, = [e for e in events if e["event_type"] == "station_status"
                   and e["payload"]["state"] == "closed"]
        assert closed["payload"]["error"] == error
        assert manager.poll_events() == ()
    finally:
        manager.close(timeout=0.2)


@pytest.mark.parametrize("target", [crash_worker, exit_before_ready])
def test_unexpected_exit_is_reported_and_resources_released(task, target):
    manager = AnalysisProcessManager(task.station_id, worker_target=target)
    try:
        pid = manager.start()
        if target is crash_worker:
            wait_ready(manager)
            manager.submit(task)
        events = wait_for(manager, lambda events: manager.closed)
        assert any(e["event_type"] == "station_status" and e["payload"]["state"] == "failed"
                   for e in events)
        results = [e for e in events if e["event_type"] == "task_result"]
        assert len(results) == (1 if target is crash_worker else 0)
        if results:
            assert results[0]["payload"]["execution_status"] == "分析失败"
            assert results[0]["payload"]["error"]["type"] == "WorkerExit"
        assert manager.poll_events() == ()
        assert pid not in {child.pid for child in multiprocessing.active_children()}
    finally:
        manager.close(timeout=0.2)


def test_hung_worker_is_terminated_on_bounded_close(task):
    manager = AnalysisProcessManager(task.station_id, worker_target=hang_worker)
    try:
        pid = manager.start()
        wait_ready(manager)
        manager.submit(task)
        started = time.monotonic()
        events = manager.close(timeout=0.2, terminate_timeout=3)
        assert time.monotonic() - started < 5
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["error"]["type"] == "ShutdownTimeout"
        assert manager.closed
        assert pid not in {child.pid for child in multiprocessing.active_children()}
    finally:
        manager.close(timeout=0.2)


def test_management_import_is_light():
    script = """
import sys
from base.analysis.process_manager import AnalysisProcessManager
for prefix in ('ui', 'PyQt5', 'numpy', 'scipy', 'soundfile', 'matplotlib', 'base.analysis.executor'):
    assert not any(k == prefix or k.startswith(prefix + '.') for k in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def never_ready(connection, station_id, generation):
    while True:
        time.sleep(1)


def partial_result_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    connection.recv()
    # A large pickle cannot fit in a pipe buffer. Exit midway through writing it.
    from threading import Thread

    def abort():
        time.sleep(0.02)
        os._exit(23)

    Thread(target=abort, daemon=True).start()
    connection.send_bytes(b"x" * (32 * 1024 * 1024))
    os._exit(23)


def wrong_result_worker(connection, station_id, generation):
    connection.send(WorkerEvent("ready", station_id, generation))
    command = connection.recv()
    connection.send(WorkerEvent("task_result", station_id, generation, command.task.task_id, {
        "task_id": command.task.task_id, "station_id": "wrong-station",
        "condition_id": command.task.condition_id,
    }))
    connection.recv()


def stale_event_worker(connection, station_id, generation):
    from base.analysis.worker import analysis_worker_main

    connection.send(WorkerEvent("worker_error", station_id, generation - 1,
                                payload={"stage": "old", "type": "Old", "message": "stale"}))
    analysis_worker_main(connection, station_id, generation)


def exception_once_worker(connection, station_id, generation):
    from base.analysis import executor
    from base.analysis.worker import analysis_worker_main

    original = executor.execute_task
    first = True

    def execute(task):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("synthetic uncaught executor failure")
        return original(task)

    executor.execute_task = execute
    analysis_worker_main(connection, station_id, generation)


def test_uncaught_executor_error_still_allows_next_real_wav(task):
    manager = AnalysisProcessManager(task.station_id, worker_target=exception_once_worker)
    try:
        pid = manager.start()
        wait_ready(manager)
        manager.submit(task)
        failed, _ = wait_result(manager, task.task_id)
        assert failed["error"]["type"] == "RuntimeError"
        second = replace(task, task_id="next-real-task")
        manager.submit(second)
        result, _ = wait_result(manager, second.task_id)
        assert result["execution_status"] == "分析完成" and manager.pid == pid
    finally:
        manager.close()


def test_partial_ipc_or_invalid_result_cannot_hang_poll_or_duplicate_result(task):
    for target in (partial_result_worker, wrong_result_worker):
        manager = AnalysisProcessManager(task.station_id, worker_target=target)
        try:
            manager.start()
            wait_ready(manager)
            manager.submit(task)
            events = []
            deadline = time.monotonic() + 15
            while not manager.closed and time.monotonic() < deadline:
                start = time.monotonic()
                events.extend(manager.poll_events())
                assert time.monotonic() - start < 1.0
                time.sleep(0.01)
            assert manager.closed
            results = [e["payload"] for e in events if e["event_type"] == "task_result"]
            assert len(results) == 1
            assert results[0]["execution_status"] == "分析失败"
            assert manager.poll_events() == ()
        finally:
            manager.close(timeout=0.2)


def test_startup_timeout_is_bounded_and_has_no_fictitious_task_result(task):
    manager = AnalysisProcessManager(task.station_id, startup_timeout=0.1, worker_target=never_ready)
    try:
        manager.start()
        events = wait_for(manager, lambda events: manager.closed, timeout=10)
        assert not any(e["event_type"] == "task_result" for e in events)
        failures = [e["payload"]["error"] for e in events
                    if e["event_type"] == "station_status" and e["payload"]["state"] == "failed"]
        assert len(failures) == 1 and failures[0]["type"] == "StartupTimeout"
    finally:
        manager.close(timeout=0.2)


def test_old_generation_ignored_and_explicit_restart_releases_old_pid(task):
    manager = AnalysisProcessManager(task.station_id, worker_target=stale_event_worker)
    try:
        manager.start()
        assert not any(e["payload"].get("error") for e in wait_ready(manager))
        first_pid = manager.pid
        manager.close()
        manager.start()
        wait_ready(manager)
        manager.submit(task)
        result, _ = wait_result(manager, task.task_id)
        assert result["execution_status"] == "分析完成"
        assert manager.pid != first_pid
        assert first_pid not in {p.pid for p in multiprocessing.active_children()}
    finally:
        manager.close()


def test_unserializable_input_is_not_accepted_and_next_task_works(task):
    manager = AnalysisProcessManager(task.station_id)
    try:
        manager.start()
        wait_ready(manager)
        broken = replace(task, instances=(replace(task.instances[0], parameters={"bad": lambda: None}),))
        with pytest.raises((pickle.PicklingError, AttributeError)):
            manager.submit(broken)
        assert manager.state == "ready"
        manager.submit(task)
        assert wait_result(manager, task.task_id)[0]["execution_status"] == "分析完成"
    finally:
        manager.close()


def test_process_creation_failure_closes_pipe_endpoints(task, monkeypatch):
    manager = AnalysisProcessManager(task.station_id)
    context = manager._context
    handles = []
    original = context.Pipe

    def pipe(*args, **kwargs):
        pair = original(*args, **kwargs)
        handles.extend(pair)
        return pair

    def fail_process(**kwargs):
        raise OSError("synthetic process creation failure")

    monkeypatch.setattr(context, "Pipe", pipe)
    monkeypatch.setattr(context, "Process", fail_process)
    with pytest.raises(OSError, match="creation failure"):
        manager.start()
    assert handles and all(handle.closed for handle in handles)
    assert manager.closed
    events = manager.poll_events()
    assert any(e["payload"].get("error", {}).get("type") == "OSError"
               for e in events if e["payload"].get("error"))


def test_undelivered_result_prevents_another_dispatch(task):
    manager = AnalysisProcessManager(task.station_id)
    try:
        manager.start()
        wait_ready(manager)
        manager.submit(task)
        returned = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            event_batch = manager.poll_events(max_events=1)
            returned.extend(event_batch)
            if any(e["event_type"] == "task_result" for e in returned):
                break
            with pytest.raises(RuntimeError):
                manager.submit(replace(task, task_id="must-not-be-accepted"))
            time.sleep(0.01)
        assert any(e["event_type"] == "task_result" for e in returned)
        assert manager.state == "ready"
    finally:
        manager.close()


def test_blocked_large_command_send_is_released_when_worker_is_killed(task):
    manager = AnalysisProcessManager(task.station_id, worker_target=hang_worker)
    try:
        pid = manager.start()
        wait_ready(manager)
        large = replace(task.instances[0], parameters={"payload": "x" * (8 * 1024 * 1024)})
        manager.submit(replace(task, instances=(large,)))
        events = manager.close(timeout=0.2, terminate_timeout=3)
        assert manager.closed
        assert len([e for e in events if e["event_type"] == "task_result"]) == 1
        assert f"analysis-io-{pid}" not in {t.name for t in threading.enumerate()}
    finally:
        manager.close(timeout=0.2)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle-transfer regression")
def test_early_startup_kills_do_not_leak_parent_handles(task):
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]

    def count():
        value = ctypes.c_ulong()
        assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(value))
        return value.value

    before = count()
    manager = AnalysisProcessManager(task.station_id, startup_timeout=0, worker_target=never_ready)
    for _ in range(5):
        try:
            pid = manager.start()
            wait_for(manager, lambda events: manager.closed, timeout=10)
            assert f"analysis-io-{pid}" not in {t.name for t in threading.enumerate()}
        finally:
            manager.close(timeout=0.2)
    assert count() <= before + 2


def test_close_is_idempotent_and_empty_task_can_use_resident_worker(task):
    manager = AnalysisProcessManager(task.station_id)
    assert manager.close() == ()
    try:
        manager.start()
        wait_ready(manager)
        manager.submit(replace(task, instances=(), wav_path="missing.wav"))
        result, _ = wait_result(manager, task.task_id)
        assert result["execution_status"] == "分析完成" and result["instance_results"] == []
        manager.request_stop()
        manager.request_stop()
        wait_for(manager, lambda events: manager.closed)
        assert manager.close() == ()
    finally:
        manager.close(timeout=0.2)
