"""Serializable contracts shared by the analysis owner and worker process."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

from base.recording_process_protocol import FrozenConfig


SUPPORTED_ANALYSIS_TYPES = frozenset({"SPL", "Spec", "FBA", "AI", "FFT"})
TASK_SOURCES = frozenset({"自动分析", "手动查看"})
INSTANCE_EXECUTION_STATUSES = frozenset({"分析完成", "分析失败"})
TASK_EXECUTION_STATUSES = frozenset({"分析完成", "结果不完整", "分析失败"})
JUDGEMENT_STATUSES = frozenset({"已判定", "未产生判定", "不可用"})
ARTIFACT_STATUSES = frozenset({"已保存", "保存失败", "不保存"})
JUDGEMENTS = frozenset({"OK", "NG"})
_RUNTIME_CHANNEL_SEPARATOR = "--通道"


def _required_text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _freeze_mapping(name, value):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return FrozenConfig.snapshot(value)


def build_runtime_key(config_key: str, raw_channel: int) -> str:
    normalized_key = _required_text("config_key", config_key)
    channel = _integer("raw_channel", raw_channel)
    return f"{normalized_key}{_RUNTIME_CHANNEL_SEPARATOR}{channel + 1}"


def parse_runtime_key(runtime_key: str) -> tuple[str, int]:
    normalized = _required_text("runtime_key", runtime_key)
    config_key, separator, channel_text = normalized.rpartition(
        _RUNTIME_CHANNEL_SEPARATOR
    )
    if not separator or not config_key.strip() or not channel_text.isdecimal():
        raise ValueError("runtime_key must use config_key--通道N")
    channel_number = int(channel_text)
    if channel_number < 1:
        raise ValueError("runtime_key channel number must be >= 1")
    normalized_config_key = config_key.strip()
    if build_runtime_key(normalized_config_key, channel_number - 1) != normalized:
        raise ValueError("runtime_key is not canonical")
    return normalized_config_key, channel_number - 1


@dataclass(frozen=True)
class AnalysisChannelMapping:
    raw_channel: int
    source_wav_column: int

    def __post_init__(self):
        _integer("raw_channel", self.raw_channel)
        _integer("source_wav_column", self.source_wav_column)


@dataclass(frozen=True)
class AnalysisInstanceRequest:
    config_key: str
    runtime_key: str
    analysis_type: str
    raw_channel: int
    source_wav_column: int
    v2pa_factor: float
    parameters: Mapping
    calibration_available: bool = True

    def __post_init__(self):
        config_key = _required_text("config_key", self.config_key)
        runtime_key = _required_text("runtime_key", self.runtime_key)
        analysis_type = _required_text("analysis_type", self.analysis_type)
        _integer("raw_channel", self.raw_channel)
        _integer("source_wav_column", self.source_wav_column)
        if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(f"unsupported analysis_type: {analysis_type}")
        if runtime_key != build_runtime_key(config_key, self.raw_channel):
            raise ValueError("runtime_key does not match config_key and raw_channel")
        if isinstance(self.v2pa_factor, bool) or not isinstance(
            self.v2pa_factor,
            (int, float),
        ):
            raise ValueError("v2pa_factor must be numeric")
        factor = float(self.v2pa_factor)
        if factor <= 0.0:
            raise ValueError("v2pa_factor must be greater than zero")
        if not isinstance(self.calibration_available, bool):
            raise ValueError("calibration_available must be a boolean")
        object.__setattr__(self, "config_key", config_key)
        object.__setattr__(self, "runtime_key", runtime_key)
        object.__setattr__(self, "analysis_type", analysis_type)
        object.__setattr__(self, "v2pa_factor", factor)
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping("parameters", self.parameters),
        )


@dataclass(frozen=True)
class AnalysisTaskRequest:
    task_id: str
    condition_key: str
    wav_path: str
    source: str
    channel_mapping: tuple[AnalysisChannelMapping, ...]
    sequence_config_snapshot: Mapping
    analysis_config_snapshot: Mapping
    storage_snapshot: Mapping
    instances: tuple[AnalysisInstanceRequest, ...]

    def __post_init__(self):
        object.__setattr__(self, "task_id", _required_text("task_id", self.task_id))
        object.__setattr__(
            self,
            "condition_key",
            _required_text("condition_key", self.condition_key),
        )
        wav_path = _required_text("wav_path", self.wav_path)
        if not os.path.isabs(wav_path):
            raise ValueError("wav_path must be absolute")
        if self.source not in TASK_SOURCES:
            raise ValueError(f"unsupported task source: {self.source}")

        mappings = tuple(self.channel_mapping)
        if not mappings or any(
            not isinstance(item, AnalysisChannelMapping) for item in mappings
        ):
            raise ValueError("channel_mapping must contain channel mappings")
        by_raw_channel = {item.raw_channel: item.source_wav_column for item in mappings}
        if len(by_raw_channel) != len(mappings) or len(
            {item.source_wav_column for item in mappings}
        ) != len(mappings):
            raise ValueError("channel mapping must be unique")
        if tuple(sorted(item.source_wav_column for item in mappings)) != tuple(
            range(len(mappings))
        ):
            raise ValueError("channel mapping must cover every WAV column")

        instances = tuple(self.instances)
        if not instances or any(
            not isinstance(item, AnalysisInstanceRequest) for item in instances
        ):
            raise ValueError("instances must contain analysis requests")
        runtime_keys = [item.runtime_key for item in instances]
        if len(runtime_keys) != len(set(runtime_keys)):
            raise ValueError("runtime_key values must be unique")
        for instance in instances:
            if by_raw_channel.get(instance.raw_channel) != instance.source_wav_column:
                raise ValueError("instance does not match channel mapping")

        object.__setattr__(self, "wav_path", os.path.abspath(wav_path))
        object.__setattr__(self, "channel_mapping", mappings)
        object.__setattr__(self, "instances", instances)
        for name in (
            "sequence_config_snapshot",
            "analysis_config_snapshot",
            "storage_snapshot",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_mapping(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class AnalysisProgress:
    task_id: str
    stage: str
    completed_instances: int
    total_instances: int
    message: str = ""
    runtime_key: str = ""

    def __post_init__(self):
        _required_text("task_id", self.task_id)
        _required_text("stage", self.stage)
        completed = _integer("completed_instances", self.completed_instances)
        total = _integer("total_instances", self.total_instances, 1)
        if completed > total:
            raise ValueError("completed_instances cannot exceed total_instances")


@dataclass(frozen=True)
class AnalysisArtifactResult:
    kind: str
    status: str
    path: str = ""
    error_message: str = ""

    def __post_init__(self):
        _required_text("kind", self.kind)
        if self.status not in ARTIFACT_STATUSES:
            raise ValueError("unsupported artifact status")
        if self.status == "已保存" and not str(self.path or "").strip():
            raise ValueError("saved artifact requires a path")


@dataclass(frozen=True)
class AnalysisInstanceResult:
    task_id: str
    config_key: str
    runtime_key: str
    analysis_type: str
    raw_channel: int
    source_wav_column: int
    execution_status: str
    contributes_to_final: bool
    judgement: str | None
    metrics: Mapping
    display_payload: Mapping
    artifacts: tuple[AnalysisArtifactResult, ...] = ()
    error_stage: str = ""
    error_type: str = ""
    error_message: str = ""

    def __post_init__(self):
        _required_text("task_id", self.task_id)
        config_key = _required_text("config_key", self.config_key)
        runtime_key = _required_text("runtime_key", self.runtime_key)
        if self.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(f"unsupported analysis_type: {self.analysis_type}")
        _integer("raw_channel", self.raw_channel)
        _integer("source_wav_column", self.source_wav_column)
        if runtime_key != build_runtime_key(config_key, self.raw_channel):
            raise ValueError("runtime_key does not match result identity")
        if self.execution_status not in INSTANCE_EXECUTION_STATUSES:
            raise ValueError("unsupported execution_status")
        if not isinstance(self.contributes_to_final, bool):
            raise ValueError("contributes_to_final must be a boolean")
        if self.execution_status != "分析完成" and self.judgement is not None:
            raise ValueError("failed execution cannot carry a judgement")
        if self.contributes_to_final and self.execution_status == "分析完成":
            if self.judgement not in JUDGEMENTS:
                raise ValueError("completed contributing result needs OK/NG")
        elif self.judgement is not None:
            raise ValueError("non-contributing result cannot carry a judgement")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, AnalysisArtifactResult) for item in artifacts):
            raise ValueError("artifacts must contain artifact results")
        object.__setattr__(self, "metrics", _freeze_mapping("metrics", self.metrics))
        object.__setattr__(
            self,
            "display_payload",
            _freeze_mapping("display_payload", self.display_payload),
        )
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True)
class AnalysisTaskResult:
    task_id: str
    condition_key: str
    wav_path: str
    source: str
    execution_status: str
    judgement_status: str
    final_judgement: str | None
    instance_results: tuple[AnalysisInstanceResult, ...]
    error_stage: str = ""
    error_type: str = ""
    error_message: str = ""

    def __post_init__(self):
        _required_text("task_id", self.task_id)
        _required_text("condition_key", self.condition_key)
        if not os.path.isabs(self.wav_path):
            raise ValueError("wav_path must be absolute")
        if self.source not in TASK_SOURCES:
            raise ValueError("unsupported task source")
        if self.execution_status not in TASK_EXECUTION_STATUSES:
            raise ValueError("unsupported execution_status")
        if self.judgement_status not in JUDGEMENT_STATUSES:
            raise ValueError("unsupported judgement_status")
        if self.final_judgement is not None and self.final_judgement not in JUDGEMENTS:
            raise ValueError("final_judgement must be OK, NG or empty")
        if self.judgement_status != "已判定" and self.final_judgement is not None:
            raise ValueError("only judged tasks can carry final_judgement")
        results = tuple(self.instance_results)
        for result in results:
            if not isinstance(result, AnalysisInstanceResult):
                raise ValueError("instance_results must contain analysis results")
            if result.task_id != self.task_id:
                raise ValueError("instance result task_id does not match task")
        object.__setattr__(self, "instance_results", results)


@dataclass(frozen=True)
class AnalysisWorkerFailure:
    task_id: str
    stage: str
    error_type: str
    message: str
    traceback_text: str = ""

    def __post_init__(self):
        _required_text("task_id", self.task_id)
        _required_text("stage", self.stage)
        _required_text("error_type", self.error_type)
        _required_text("message", self.message)
