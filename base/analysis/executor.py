"""Execute a task; the worker owns its process and save-thread lifecycle."""

from copy import deepcopy
from concurrent.futures import wait
import logging

import numpy as np

from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis_config_validation import ai_execution_issue
from base.analysis_compute_context import TaskComputeContext
from base.loudness_analysis_adapter import loudness_limit_data

from .contracts import AnalysisTask
from .evaluation import evaluate_instance
from .results import (
    calculation_payload, error_record, instance_result, plain_data, task_result,
)
from .storage import save_instance_artifacts
from .wav_input import load_task_wav, resolve_calibration, sample_window


_LOGGER = logging.getLogger(__name__)
_SUPPORTED_TYPES = frozenset({"SPL", "FFT", "FBA", "Spec", "LOUD", "SHRP", "ROUGH"})


def execute_task(task: AnalysisTask, *, reuse_computations: bool = True,
                 on_judgement_ready=None, save_executor=None) -> dict:
    """Return a fixed result, without UI, current-config queries or processes.

    The caller must publish/close the WAV before calling. Preparation issues do
    not replace execution checks: calibration may only be known after reading.
    """
    if task.segmented_analysis.get("mode", "none") != "none":
        return task_result(task, [], error_record(
            "validation", "当前执行器尚不支持分段分析", error_type="UnsupportedSegmentation",
        ))
    if not task.instances:
        return task_result(task, [])
    try:
        samples, sample_rate, metadata = load_task_wav(task)
    except Exception as error:
        # File/library boundary; preserve diagnostics and return one task failure.
        _LOGGER.exception("Cannot read WAV for analysis task %s", task.task_id)
        return task_result(task, [], error_record("wav_read", error))
    context = TaskComputeContext() if reuse_computations else None
    pending = []
    last_required = max((i for i, item in enumerate(task.instances)
                         if item.parameters.get("limit_checked", False)), default=-1)
    try:
        results = []
        for index, instance in enumerate(task.instances):
            result = _execute_instance(task, instance, samples, sample_rate, metadata, context)
            results.append(result)
            if index == last_required and on_judgement_ready is not None:
                fields = ("config_key", "analysis_type", "channel_index", "execution_status",
                          "judgement_required", "judgement_status", "judgement", "error")
                on_judgement_ready({
                    "task_id": task.task_id, "station_id": task.station_id,
                    "condition_id": task.condition_id,
                    "instance_results": [{key: deepcopy(item[key]) for key in fields}
                                         for item in results if item["judgement_required"]],
                })
            if result["execution_status"] != "分析完成" or not task.persist_artifacts:
                continue
            if save_executor is None:
                result["artifacts"] = save_instance_artifacts(task, instance, result)
            else:
                pending.append((result, save_executor.submit(
                    save_instance_artifacts, task, instance, result,
                )))
        for result, future in pending:
            result["artifacts"] = future.result()
        return plain_data(task_result(task, results))
    finally:
        # Also drain on calculation/callback/submission failure: no writes may
        # escape into the next task of this resident worker.
        if pending:
            wait([future for _, future in pending])
            pending.clear()
        if context is not None:
            context.clear()
            _LOGGER.debug(
                "Task computation reuse %s: hits=%s misses=%s computes=%s lookup_s=%.6f compute_s=%.6f",
                task.task_id, context.hits, context.misses, context.compute_count,
                context.lookup_seconds, context.compute_seconds,
            )


def _execute_instance(task, instance, samples, sample_rate, metadata, compute_context=None):
    result = instance_result(instance)
    if instance.analysis_type == "AI":
        kind, message = ai_execution_issue(instance.parameters)
        result["error"] = error_record("validation", message, error_type=kind)
        return result
    if instance.analysis_type not in _SUPPORTED_TYPES:
        result["error"] = error_record(
            "validation", f"不支持的分析类型：{instance.analysis_type}",
            error_type="UnsupportedAnalysisType",
        )
        return result
    stage = "validation"
    try:
        index = instance.channel_index
        if type(index) is not int or not 0 <= index < len(task.channels):
            raise ValueError("分析实例的 WAV 通道索引越界")
        channel = task.channels[index]
        if channel.channel_index != index:
            raise ValueError("channels 顺序与 WAV 通道索引不一致")
        if channel.signal_type != "sound_pressure":
            result["error"] = error_record(
                stage, f"{instance.analysis_type} 尚不支持信号类型 {channel.signal_type}",
                error_type="UnsupportedSignalType",
            )
            return result
        time_config = (instance.parameters.get("advanced", {})
                       if instance.analysis_type in {"LOUD", "SHRP", "ROUGH"} else instance.parameters)
        first, stop = sample_window(time_config, len(samples), sample_rate)
        # SPL weighting runs before cropping, so invalid earlier samples matter.
        used = (
            samples[:stop, index] if instance.analysis_type == "SPL"
            else samples[first:stop, index]
        )
        if not np.all(np.isfinite(used)):
            raise ValueError("分析通道包含非有限音频样本")
        stage = "calibration"
        factor, calibration_source = resolve_calibration(channel, metadata)
        stage = "calculation"
        calculation = calculate_analysis_instance(
            instance.analysis_type, samples[:, index], sample_rate,
            deepcopy(instance.parameters), factor, source="station_analysis",
            sequence_snapshot={"segment_window_samples": (first, stop)},
            evaluate_limits=False, full_spectrogram=True,
            compute_context=compute_context,
            input_identity=(task.task_id, index, first, stop, "wav_voltage_float32"),
        )
        metrics, payload = calculation_payload(instance, calculation, first / sample_rate)
        for key in ("overall_spl", "overall_db", "overall_weighted_db", "peak_value"):
            if key in metrics and (metrics[key] is None or not np.isfinite(metrics[key])):
                raise ValueError(f"计算指标 {key} 不是有限数值")
        metrics.update(
            sample_rate=sample_rate, wav_duration_sec=len(samples) / sample_rate,
            analysis_start_time_sec=first / sample_rate,
            analysis_end_time_sec=stop / sample_rate,
            v2pa_factor=factor, calibration_source=calibration_source,
        )
        result.update(execution_status="分析完成", metrics=metrics, display_payload=payload)
    except Exception as error:
        # An algorithm failure must not discard another item's completed data.
        _LOGGER.exception(
            "Analysis instance failed: %s/%s/CH%s",
            task.task_id, instance.config_key, instance.channel_index,
        )
        result["error"] = error_record(stage, error)
        return result
    _judge(instance, calculation, result)
    return result


def _judge(instance, calculation, result):
    if not instance.parameters.get("limit_checked", False):
        result["judgement_status"] = "not_enabled"
        return
    try:
        evaluation = evaluate_instance(instance, calculation)
        if instance.analysis_type in {"LOUD", "SHRP", "ROUGH"}:
            result["metrics"].update(
                limit_metric=instance.parameters.get("limit_metric", "curve_y"),
                limit_unit=evaluation["unit"],
            )
        if evaluation["scalar"]:
            for side in ("upper", "lower"):
                value = evaluation[side][0]
                result["metrics"][f"overall_{side}_limit"] = (
                    float(value) if np.isfinite(value) else None
                )
        else:
            if instance.analysis_type == "LOUD":
                result["display_payload"]["limit_unit"] = evaluation["unit"]
            if (instance.analysis_type != "LOUD"
                    or evaluation["unit"] == result["display_payload"]["y_unit"]):
                for side in ("upper", "lower"):
                    result["display_payload"][side] = evaluation[side]
                if instance.analysis_type == "LOUD":
                    # Resolve only the visual bounds on the legacy display geometry.
                    # The verdict and exported samples above still use full raw data.
                    presentation = result["display_payload"]["plot_curve"]
                    _, _, upper, lower, _, _ = loudness_limit_data(instance.parameters, {
                        **calculation, "curve": presentation,
                        "loudness_curves": calculation["plot_loudness_curves"],
                    })
                    presentation.update(upper=upper, lower=lower)
        result.update(
            judgement_status="completed", judgement=evaluation["judgement"],
            violations=evaluation["violations"],
        )
    except Exception as error:
        _LOGGER.exception("Judgement failed: %s/CH%s", instance.config_key, instance.channel_index)
        result.update(judgement_status="failed", error=error_record("judgement", error))
