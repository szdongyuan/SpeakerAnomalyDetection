"""Private, picklable worker messages; no numerical or UI imports."""

from dataclasses import dataclass, field

from .contracts import AnalysisTask


@dataclass(frozen=True)
class RunTask:
    task: AnalysisTask


@dataclass(frozen=True)
class StopWorker:
    pass


@dataclass(frozen=True)
class WorkerEvent:
    kind: str
    station_id: str
    generation: int
    task_id: str | None = None
    payload: dict = field(default_factory=dict)


def runtime_error(stage, kind, message):
    return {"stage": stage, "type": kind, "message": str(message)}


def failed_task(task, error):
    """A transport/worker failure has no recoverable per-instance payload."""
    return {
        "task_id": task.task_id, "station_id": task.station_id,
        "condition_id": task.condition_id, "execution_status": "分析失败",
        "instance_results": [], "error": error,
    }
