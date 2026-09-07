import time

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
