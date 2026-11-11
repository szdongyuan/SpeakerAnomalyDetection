"""Execute one prepared task; process ownership belongs to the later service."""

from copy import deepcopy
import logging

import numpy as np

from base.analysis_algorithm_adapters import calculate_analysis_instance

from .contracts import AnalysisTask
from .evaluation import evaluate_instance
from .results import (
    calculation_payload, error_record, instance_result, plain_data, task_result,
)
from .storage import save_curve, save_loudness_summary, save_overall_spl
from .wav_input import load_task_wav, resolve_calibration, sample_window


_LOGGER = logging.getLogger(__name__)
_SUPPORTED_TYPES = frozenset({"SPL", "FFT", "FBA", "Spec", "LOUD"})


def execute_task(task: AnalysisTask) -> dict:
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
    results = [
        _execute_instance(task, instance, samples, sample_rate, metadata)
        for instance in task.instances
    ]
    return plain_data(task_result(task, results))


def _execute_instance(task, instance, samples, sample_rate, metadata):
    result = instance_result(instance)
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
                       if instance.analysis_type == "LOUD" else instance.parameters)
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
    exports = []
    if instance.analysis_type == "SPL":
        exports.append(("overall_spl_csv", save_overall_spl))
    elif instance.analysis_type == "LOUD":
        exports.append(("loudness_summary_csv", save_loudness_summary))
    if instance.analysis_type in {"SPL", "FFT", "FBA", "LOUD"}:
        exports.append(("curve_csv", save_curve))
    for kind, save in exports:
        artifact = {"kind": kind, "status": "failed", "path": None, "error": None}
        try:
            artifact["path"] = save(task, instance, result)
            artifact["status"] = "saved"
        except Exception as error:
            _LOGGER.exception("Cannot save %s for %s/%s", kind, task.task_id, instance.config_key)
            artifact["error"] = error_record("saving", error)
        result["artifacts"].append(artifact)
    return result


def _judge(instance, calculation, result):
    if not instance.parameters.get("limit_checked", False):
        result["judgement_status"] = "not_enabled"
        return
    try:
        evaluation = evaluate_instance(instance, calculation)
        if instance.analysis_type == "LOUD":
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
        result.update(
            judgement_status="completed", judgement=evaluation["judgement"],
            violations=evaluation["violations"],
        )
    except Exception as error:
        _LOGGER.exception("Judgement failed: %s/CH%s", instance.config_key, instance.channel_index)
        result.update(judgement_status="failed", error=error_record("judgement", error))
