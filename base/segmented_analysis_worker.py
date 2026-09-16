"""Segment orchestration inside the existing one-WAV worker process."""

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import time
import traceback

import numpy as np

from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis_artifact_paths import (
    build_csv_path,
    format_channel_name,
    storage_context_from_metadata,
)
from base.analysis_csv_exporter import _write_csv_atomic, _csv_number
from base.analysis_process_protocol import (
    AnalysisArtifactResult,
    AnalysisInstanceResult,
    AnalysisProgress,
    AnalysisSegmentResult,
)
from base.analysis_result_summary import summarize_analysis_task
from base.spl_csv_schema import overall_spl_csv_columns


class _WholeProgress:
    def __init__(self, queue, total):
        self.queue, self.total = queue, total

    def put(self, event):
        name, value = event
        if name == "progress":
            value = replace(
                value, total_instances=self.total,
                message="整段概览：" + value.message,
            )
        self.queue.put((name, value))


def execute_segmented_task(request, audio, sample_rate, event_queue, log_queue):
    from base.analysis_worker import _execute_task, _emit_log

    if any(s.sample_rate != sample_rate for s in request.segment_plan):
        raise ValueError("WAV 采样率已变化，请重新创建分析任务")
    total = len(request.instances) * (1 + len(request.segment_plan))
    whole = _execute_task(
        request, audio, sample_rate, _WholeProgress(event_queue, total), log_queue,
    )
    segments = []
    completed = len(request.instances)
    for segment in request.segment_plan:
        results = []
        sequence = request.sequence_config_snapshot.to_dict()
        sequence["segment_window_samples"] = (
            segment.window_start_sample, segment.window_end_sample,
        )
        for instance in request.instances:
            results.append(_execute_segment_instance(
                request, instance, segment, audio, sequence, log_queue,
            ))
            completed += 1
            event_queue.put(("progress", AnalysisProgress(
                request.task_id, "分段分析", completed, total,
                f"{segment.label}：{instance.runtime_key}", instance.runtime_key,
            )))
        summary = summarize_analysis_task(request, results)
        segments.append(AnalysisSegmentResult(
            segment, tuple(results), summary.execution_status,
            summary.judgement_status, summary.final_judgement,
        ))
    summary = summarize_analysis_task(
        request, [item for segment in segments for item in segment.instance_results],
    )
    summary = replace(
        summary, instance_results=whole.instance_results, segments=tuple(segments),
        condition_snapshot=request.condition_snapshot,
        channel_labels=request.storage_snapshot.to_dict().get("channel_labels", {}),
    )
    if len(audio) < request.segment_plan[-1].end_sample:
        summary = replace(
            summary, execution_status="结果不完整", judgement_status="不可用",
            final_judgement=None, error_stage="录音完整性",
            error_message="录音长度短于计划时长",
        )
        _emit_log(log_queue, "ERROR", "analysis_recording_incomplete", request,
                  actual_samples=len(audio), planned_samples=request.segment_plan[-1].end_sample,
                  error_message=summary.error_message)
    if request.source == "自动分析":
        summary = _with_artifacts(summary, _save_segment_csvs(request, summary))
        for item in summary.instance_results:
            for saved in item.artifacts:
                if saved.status == "保存失败":
                    _emit_log(log_queue, "ERROR", "analysis_artifact_save_failed", request,
                              artifact_kind=saved.kind, error_message=saved.error_message)
    return summary


def _execute_segment_instance(request, instance, segment, audio, sequence, log_queue=None):
    from base.analysis_worker import _contributes_to_final, _plain_mapping, _emit_log

    config = instance.parameters.to_dict()
    contributes = _contributes_to_final(instance.analysis_type, config)
    identity = dict(
        task_id=request.task_id, config_key=instance.config_key,
        runtime_key=instance.runtime_key, analysis_type=instance.analysis_type,
        raw_channel=instance.raw_channel, source_wav_column=instance.source_wav_column,
        contributes_to_final=contributes, display_payload={},
    )
    calculation = None
    started = time.monotonic()
    log_fields = dict(
        segment_index=segment.segment_index, segment_label=segment.label,
        window_start_seconds=segment.window_start_sample / segment.sample_rate,
        window_end_seconds=segment.window_end_sample / segment.sample_rate,
        config_key=instance.config_key, runtime_key=instance.runtime_key,
        analysis_type=instance.analysis_type, raw_channel=instance.raw_channel,
    )
    _emit_log(log_queue, "INFO", "analysis_segment_instance_started", request, **log_fields)
    try:
        if not segment.available(len(audio)):
            raise ValueError("录音不足，无法覆盖本段完整分析窗口")
        calculation = calculate_analysis_instance(
            instance.analysis_type, audio[:, instance.source_wav_column],
            segment.sample_rate, config, instance.v2pa_factor,
            source=request.source, sequence_snapshot=sequence,
        )
        judgement = calculation.get("judgement") if contributes else None
        if contributes and judgement not in {"OK", "NG"}:
            raise ValueError("该分析项未产生必需的 OK/NG 判定")
        metrics = _plain_mapping(calculation.get("metrics") or {})
        if config.get("limit_checked"):
            metrics.update(_compact_curve_limits(calculation.get("curve") or {}))
        _emit_log(log_queue, "INFO", "analysis_segment_instance_finished", request,
                  **log_fields, duration_seconds=round(time.monotonic() - started, 3))
        return AnalysisInstanceResult(
            **identity, execution_status="分析完成", judgement=judgement, metrics=metrics,
        )
    except Exception as error:
        # A failed item is a result; the enclosing loop still runs later items/segments.
        _emit_log(log_queue, "ERROR", "analysis_segment_instance_failed", request,
                  **log_fields, error_type=type(error).__name__, error_message=str(error),
                  traceback_text=traceback.format_exc(),
                  duration_seconds=round(time.monotonic() - started, 3))
        return AnalysisInstanceResult(
            **identity, execution_status="分析失败", judgement=None, metrics={},
            error_stage="分段分析", error_type=type(error).__name__, error_message=str(error),
        )
    finally:
        if calculation is not None:
            calculation.clear()


def _compact_curve_limits(curve):
    summaries = {}
    for side in ("lower", "upper"):
        values = np.asarray(curve.get(side, []), dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            summaries[f"{side}_limit_summary"] = (
                f"{low:.3f}" if np.isclose(low, high) else f"曲线 {low:.3f}～{high:.3f}"
            )
    return summaries


def _with_artifacts(result, artifacts):
    first, *rest = result.instance_results
    first = replace(first, artifacts=first.artifacts + tuple(artifacts))
    return replace(result, instance_results=(first, *rest))


def _save_segment_csvs(request, result):
    metadata = request.storage_snapshot.to_dict()
    artifacts = []
    try:
        context = storage_context_from_metadata(metadata)
    except (ValueError, TypeError) as error:
        return [AnalysisArtifactResult("CSV:总体声压级", "保存失败", error_message=str(error))]
    rows_by_key = OrderedDict()
    configs = request.analysis_config_snapshot.to_dict()
    voltage = request.condition_snapshot.get("input_voltage", "")
    for segment in result.segments:
        for item in sorted(segment.instance_results, key=lambda i: i.raw_channel):
            config = configs[item.config_key]
            if item.analysis_type != "SPL" or not (config.get("show_overall_spl") or
                    (config.get("limit_checked") and config.get("limit_metric", "overall_spl") == "overall_spl")):
                continue
            metrics = item.metrics
            judged = config.get("limit_checked") and config.get("limit_metric", "overall_spl") == "overall_spl"
            name = segment.segment.label + "_" + format_channel_name(item.raw_channel, metadata.get("channel_labels"))
            rows_by_key.setdefault(item.config_key, []).append((name, _csv_number(metrics.get("overall_spl")),
                _csv_number(metrics.get("overall_lower_limit")), _csv_number(metrics.get("overall_upper_limit")),
                item.judgement or "" if judged else "", voltage))
    for key, rows in rows_by_key.items():
        path = build_csv_path(context, Path(request.wav_path).stem, key, "总体声压级")
        try:
            columns = overall_spl_csv_columns(configs[key].get("weighting", "Z"))
            _write_csv_atomic(path, ("通道", *columns, "result", "输入电压"), rows)
            artifacts.append(AnalysisArtifactResult("CSV:总体声压级", "已保存", str(path)))
        except (OSError, ValueError, TypeError) as error:
            artifacts.append(AnalysisArtifactResult("CSV:总体声压级", "保存失败", str(path), error_message=str(error)))
    return artifacts
