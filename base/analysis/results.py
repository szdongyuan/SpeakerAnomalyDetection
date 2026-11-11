"""Fixed controller-facing result dictionaries; numerical arrays become lists."""

import math

import numpy as np

from base.pre_processing.spl_runtime_config import resolve_spl_unit


def error_record(stage, error, *, error_type=None):
    return {"stage": stage, "type": error_type or type(error).__name__, "message": str(error)}


def instance_result(instance):
    return {
        "config_key": instance.config_key, "analysis_type": instance.analysis_type,
        "channel_index": instance.channel_index, "execution_status": "分析失败",
        "judgement_status": "not_executed", "judgement": None,
        "metrics": {}, "display_payload": None, "violations": None,
        "artifacts": [], "error": None,
    }


def task_result(task, instances, error=None):
    completed = sum(item["execution_status"] == "分析完成" for item in instances)
    if error is not None or (instances and not completed):
        status = "分析失败"
    elif completed != len(instances) or any(
        item["judgement_status"] == "failed" for item in instances
    ):
        status = "结果不完整"
    else:
        status = "分析完成"
    return {
        "task_id": task.task_id, "station_id": task.station_id,
        "condition_id": task.condition_id, "execution_status": status,
        "instance_results": instances, "error": error,
    }


def calculation_payload(instance, calculation, start_time):
    metrics = dict(calculation["metrics"])
    config = instance.parameters
    kind = instance.analysis_type
    if kind == "Spec":
        plot = calculation["plot"]
        scale = str(config.get("freq_scale_type", "linear") or "linear").lower()
        return {
            "unit": "dB re 20 uPa (transform magnitude)",
            "transform": "CQT" if scale == "log" else "STFT",
            "normalization": "raw_transform_magnitude",
            "reference_pa": 20e-6, "amplitude_floor_pa": 1e-5,
        }, {
            "kind": "spectrogram", "time_s": plot["x"] + start_time,
            "frequency_hz": plot["y"], "values_db": plot["z"],
            "matrix_axes": ["frequency", "time"], "reduction": "none",
            "time_reference": "wav_start", "top_db": None,
        }
    weighting = str(metrics.get("weighting", config.get("weighting", "Z")) or "Z")
    unit = resolve_spl_unit(weighting)
    if kind == "FFT" and metrics["display_mode"] == "delta":
        unit = "dB"
    curve = calculation["curve"]
    metrics.update(weighting=weighting, unit=unit)
    payload = {
        "kind": "bands" if kind == "FBA" else "curve",
        "x": curve["x"], "y": curve["y"], "upper": [], "lower": [],
        "x_unit": "s" if kind == "SPL" else "Hz", "y_unit": unit,
        "reduction": "none",
    }
    if kind == "FFT":
        finite = np.flatnonzero(np.isfinite(curve["y"]))
        if not finite.size:
            raise ValueError("FFT 没有有限的频谱数值")
        peak = finite[np.argmax(curve["y"][finite])]
        metrics.update(
            peak_frequency_hz=float(curve["x"][peak]),
            peak_search_range_hz=[float(curve["x"][0]), float(curve["x"][-1])],
        )
        payload["baseline"] = calculation["plot"]["baseline"]
    elif kind == "FBA":
        payload["band_labels"] = calculation["plot"]["labels"]
    return metrics, payload


def plain_data(value):
    """Keep missing numerical points as null, never fabricated zero readings."""
    if isinstance(value, np.ndarray):
        return plain_data(value.tolist())
    if isinstance(value, np.generic):
        return plain_data(value.item())
    if isinstance(value, dict):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
