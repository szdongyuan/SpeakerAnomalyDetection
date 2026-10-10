import time
from types import SimpleNamespace

import pytest

from base import analysis_service
from base.log_manager import LogManager
from unit_test.logging_test_support import isolated_project_logger

from base.analysis_process_protocol import AnalysisTaskResult
from base.analysis_service import AnalysisProcessService
from unit_test.base.test_analysis_process_protocol import _task


def _successful_worker(request, event_queue, log_queue):
    log_queue.put({"level": "INFO", "event": "fake", "task_id": request.task_id})
    event_queue.put(
        (
            "result",
            AnalysisTaskResult(
                request.task_id,
                request.condition_key,
                request.wav_path,
                request.source,
                "分析完成",
                "未产生判定",
                None,
                (),
            ),
        )
    )


def _crashing_worker(_request, _event_queue, _log_queue):
    raise RuntimeError("worker boom")


def _wait_terminal(service, timeout=15):
    deadline = time.monotonic() + timeout
    events = []
    logs = []
    while time.monotonic() < deadline:
        new_events, new_logs = service.poll()
        events.extend(new_events)
        logs.extend(new_logs)
        if not service.active:
            return events, logs
        time.sleep(0.02)
    raise AssertionError("analysis service did not finish")


def test_service_uses_one_spawned_process_and_releases_it(tmp_path):
    service = AnalysisProcessService(worker_target=_successful_worker)
    request = _task(tmp_path)
    pid = service.start(request)

    events, logs = _wait_terminal(service)

    assert pid > 0
    assert [kind for kind, _payload in events] == ["result"]
    assert logs[0]["event"] == "fake"
    assert service.active is False


def test_service_reports_child_exit_without_terminal_payload(tmp_path):
    service = AnalysisProcessService(worker_target=_crashing_worker)
    service.start(_task(tmp_path))

    events, _logs = _wait_terminal(service)

    assert events[-1][0] == "failure"
    assert events[-1][1].stage == "子进程退出"
    assert "退出码=" in events[-1][1].message


@pytest.mark.parametrize("failed_stage", [
    None, "event_queue_create", "log_queue_create", "process_create", "process_start",
])
def test_startup_stage_logs_reach_project_log_and_preserve_errors(
    tmp_path, monkeypatch, failed_stage,
):
    stages = ("event_queue_create", "log_queue_create", "process_create", "process_start")
    durations = dict(zip(stages, (0.125, 0.25, 0.375, 11.0)))
    clock = [100.0]
    calls = []
    failure = OSError("startup probe")
    monkeypatch.setattr(analysis_service, "perf_counter", lambda: clock[0])

    def run_stage(stage):
        calls.append(stage)
        clock[0] += durations[stage]
        if stage == failed_stage:
            raise failure

    def queue():
        run_stage(stages[len(calls)])
        return object()

    process = SimpleNamespace(pid=4321, start=lambda: run_stage("process_start"))

    def make_process(**kwargs):
        assert kwargs["args"][0] is request
        run_stage("process_create")
        return process

    request = _task(tmp_path)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = AnalysisProcessService()
        service._context = SimpleNamespace(Queue=queue, Process=make_process)
        if failed_stage is None:
            assert service.start(request) == 4321
        else:
            with pytest.raises(OSError) as caught:
                service.start(request)
            assert caught.value is failure
        assert LogManager.flush(2)
        lines = state.path.read_text(encoding="utf-8").splitlines()
        timings = [line for line in lines if "analysis_startup_timing " in line]

    expected = stages if failed_stage is None else stages[:stages.index(failed_stage) + 1]
    assert calls == list(expected)
    assert len(timings) == 2 * len(expected)
    for index, stage in enumerate(expected):
        begin, end = timings[2 * index:2 * index + 2]
        for line in (begin, end):
            assert f"task_id={request.task_id}" in line
            assert f"source={request.source}" in line
            assert f"condition={request.condition_key}" in line
            assert f"process=parent stage={stage}" in line
        assert "event=begin" in begin
        outcome = "failed" if stage == failed_stage else "success"
        assert f"event=end seconds={durations[stage]:.6f} outcome={outcome}" in end
