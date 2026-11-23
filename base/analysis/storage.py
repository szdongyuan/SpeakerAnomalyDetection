"""Per-item/channel CSV persistence for overall SPL and complete curves."""

import hashlib
import logging
from pathlib import Path

from base.analysis_artifact_paths import sanitize_path_component
from base.analysis_csv_exporter import (
    iter_curve_csv_table, loudness_summary_csv_row, sound_quality_summary_csv_row,
)
from consts.acoustic_analysis.common_consts import SOUND_QUALITY_ANALYSES
from base.csv_writer import write_csv_atomic

from .results import error_record


def save_instance_artifacts(task, instance, result):
    """Return file outcomes; the analysis thread owns the instance result."""
    exports = []
    if instance.analysis_type == "SPL":
        exports.append(("overall_spl_csv", save_overall_spl))
    elif instance.analysis_type == "LOUD":
        exports.append(("loudness_summary_csv", save_loudness_summary))
    elif instance.analysis_type in {"SHRP", "ROUGH"}:
        exports.append(("sound_quality_summary_csv", save_sound_quality_summary))
    if instance.analysis_type in {"SPL", "FFT", "FBA", "LOUD", "SHRP", "ROUGH"}:
        exports.append(("curve_csv", save_curve))
    artifacts = []
    for kind, save in exports:
        artifact = {"kind": kind, "status": "failed", "path": None, "error": None}
        try:
            artifact.update(path=save(task, instance, result), status="saved")
        except Exception as error:
            logging.getLogger(__name__).exception(
                "Cannot save %s for %s/%s", kind, task.task_id, instance.config_key,
            )
            artifact["error"] = error_record("saving", error)
        artifacts.append(artifact)
    return artifacts


def _item_file_label(config_key):
    """Keep display names intact while making empty sanitized names writable."""
    try:
        return sanitize_path_component(config_key, max_length=100)
    except ValueError:
        # Names consisting only of punctuation have no readable file label.
        digest = hashlib.sha256(config_key.encode("utf-8")).hexdigest()[:12]
        return f"分析项_{digest}"


def _csv_path(task, instance, suffix):
    if not task.analysis_result_directory:
        raise ValueError("未提供 analysis_result_directory，无法保存分析 CSV")
    root = Path(task.analysis_result_directory)
    if not root.is_absolute():
        raise ValueError("分析结果目录必须是绝对路径")
    # The supplied directory is a result root, independent of the WAV parent.
    directory = root / "csv" / Path(task.wav_path).stem
    label = _item_file_label(instance.config_key)
    # Windows ignores case; distinct item names may also sanitize to one label.
    if any(
        other.config_key != instance.config_key
        and other.channel_index == instance.channel_index
        and _item_file_label(other.config_key).casefold() == label.casefold()
        for other in task.instances
    ):
        digest = hashlib.sha256(instance.config_key.encode("utf-8")).hexdigest()[:12]
        label = f"{label}_{digest}"
    if instance.analysis_type in SOUND_QUALITY_ANALYSES:
        return directory / f"{label}_{suffix}_CH{instance.channel_index + 1}.csv"
    return directory / f"{label}_CH{instance.channel_index + 1}_{suffix}.csv"


def _csv_unit(unit):
    """Keep the calculated unit, spelling acoustic weighting as dB(A), etc."""
    return f"dB({unit[-1]})" if unit in ("dBA", "dBB", "dBC", "dBD") else unit


def save_overall_spl(task, instance, result):
    path = _csv_path(task, instance, "总体值")
    metrics = result["metrics"]
    scalar = str(instance.parameters.get("limit_metric", "curve_y") or "curve_y").lower() == "overall_spl"
    scalar_enabled = scalar and bool(instance.parameters.get("limit_checked"))
    unit = _csv_unit(metrics["unit"])
    judgement = "未启用"
    if scalar_enabled:
        judgement = result["judgement"] if result["judgement_status"] == "completed" else "判定失败"
    header = [f"总体声压级{unit}", f"下限{unit}", f"上限{unit}", "判定结果"]
    row = [
        metrics["overall_spl"],
        metrics.get("overall_lower_limit") if scalar_enabled else None,
        metrics.get("overall_upper_limit") if scalar_enabled else None,
        judgement,
    ]
    write_csv_atomic(path, header, [row])
    return str(path)


def save_curve(task, instance, result):
    """Persist every computed point, using the same curve as the returned result."""
    payload = result["display_payload"]
    x_name, y_name = {
        "SPL": ("时间(s)", "声压级"),
        "FFT": ("频率(Hz)", "幅值"),
        "FBA": ("中心频率(Hz)", "频段声压级"),
        "LOUD": ("时间(s)", "响度"),
        **{kind: ("时间(s)", spec["label"]) for kind, spec in SOUND_QUALITY_ANALYSES.items()},
    }[instance.analysis_type]
    unit = _csv_unit(payload["y_unit"])
    _, _, rows = iter_curve_csv_table(
        instance.config_key, x_name, f"{y_name}{unit}",
        [{"raw_channel": instance.channel_index, "csv_curve": payload}], {},
    )
    header = [x_name, f"{y_name}{unit}", f"下限{unit}", f"上限{unit}"]
    suffix = f"{y_name}时间曲线" if instance.analysis_type in SOUND_QUALITY_ANALYSES else "曲线"
    path = _csv_path(task, instance, suffix)
    write_csv_atomic(path, header, rows)
    return str(path)


def save_loudness_summary(task, instance, result):
    header, row = loudness_summary_csv_row(instance.parameters, result)
    path = _csv_path(task, instance, "汇总")
    write_csv_atomic(path, header, [row])
    return str(path)


def save_sound_quality_summary(task, instance, result):
    spec = SOUND_QUALITY_ANALYSES[instance.analysis_type]
    header, row = sound_quality_summary_csv_row(instance.analysis_type, instance.parameters, result)
    path = _csv_path(task, instance, spec["label"] + "汇总")
    write_csv_atomic(path, ["通道", *header], [[f"CH{instance.channel_index + 1}", *row]])
    return str(path)
