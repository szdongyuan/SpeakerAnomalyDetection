"""Lifecycle owner for one spawned analysis process at a time."""

from __future__ import annotations

import multiprocessing
from queue import Empty

from base.analysis_process_protocol import AnalysisWorkerFailure
from base.analysis_worker import analysis_worker_main


class AnalysisProcessService:
    def __init__(self, *, worker_target=analysis_worker_main):
        self._context = multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._process = None
        self._event_queue = None
        self._log_queue = None
        self._request = None
        self._terminal_event_seen = False

    @property
    def active(self):
        return self._process is not None

    @property
    def request(self):
        return self._request

    def start(self, request):
        if self.active:
            raise RuntimeError("已有分析进程正在运行")
        self._event_queue = self._context.Queue()
        self._log_queue = self._context.Queue()
        self._request = request
        self._terminal_event_seen = False
        self._process = self._context.Process(
            target=self._worker_target,
            args=(request, self._event_queue, self._log_queue),
            name=f"analysis-{request.task_id[:8]}",
        )
        self._process.start()
        return self._process.pid

    def poll(self):
        if not self.active:
            return (), ()
        events = self._drain(self._event_queue)
        logs = self._drain(self._log_queue)
        if any(kind in {"result", "failure"} for kind, _payload in events):
            self._terminal_event_seen = True
        if not self._process.is_alive():
            self._process.join(timeout=0)
            events += self._drain(self._event_queue)
            logs += self._drain(self._log_queue)
            if any(kind in {"result", "failure"} for kind, _payload in events):
                self._terminal_event_seen = True
            exit_code = self._process.exitcode
            if not self._terminal_event_seen:
                events += (
                    (
                        "failure",
                        AnalysisWorkerFailure(
                            self._request.task_id,
                            "子进程退出",
                            "WorkerExit",
                            f"分析子进程异常退出，退出码={exit_code}",
                        ),
                    ),
                )
            self._close_process_state()
        return events, logs

    def wait(self, timeout=None):
        if not self.active:
            return (), ()
        self._process.join(timeout=timeout)
        return self.poll()

    def _close_process_state(self):
        for queue in (self._event_queue, self._log_queue):
            if queue is None:
                continue
            queue.close()
            queue.join_thread()
        self._process = None
        self._event_queue = None
        self._log_queue = None
        self._request = None
        self._terminal_event_seen = False

    @staticmethod
    def _drain(queue):
        values = []
        if queue is None:
            return ()
        while True:
            try:
                values.append(queue.get_nowait())
            except Empty:
                break
        return tuple(values)
