"""Plain, picklable input snapshots for station analysis.

The adapter validates external data. Frozen envelopes protect identity; nested
parameter dictionaries are owned by each instance and must not be shared with
the caller or another instance. These are input objects, not execution results.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisInputIssue:
    task_id: str | None
    field_path: str
    error_type: str
    message: str


class AnalysisInputError(ValueError):
    """A structural error prevents this batch from being constructed."""

    def __init__(self, issue: AnalysisInputIssue):
        self.issue = issue
        super().__init__(f"{issue.field_path}: {issue.message}")


@dataclass(frozen=True)
class AnalysisChannel:
    channel_index: int
    channel_name: str
    signal_type: str
    channel_id: str | None = None
    device_ref: str | None = None
    physical_channel_index: int | None = None
    v2pa_factor: float | None = None


@dataclass(frozen=True)
class AnalysisInstance:
    config_key: str
    analysis_type: str
    channel_index: int
    parameters: dict


@dataclass(frozen=True)
class AnalysisTask:
    task_id: str
    station_id: str
    station_name: str | None
    condition_id: str
    condition_name: str | None
    wav_path: str
    # CSV output: <analysis_result_directory>/csv/<WAV stem>/, not beside the WAV.
    analysis_result_directory: str | None
    channels: tuple[AnalysisChannel, ...]
    instances: tuple[AnalysisInstance, ...]
    segmented_analysis: dict


@dataclass(frozen=True)
class AnalysisBatch:
    """Constructed tasks plus unresolved execution prerequisites, not readiness."""

    tasks: tuple[AnalysisTask, ...]
    preparation_issues: tuple[AnalysisInputIssue, ...]
