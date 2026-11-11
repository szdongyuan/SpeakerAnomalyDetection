"""Per-item/channel CSV persistence for overall SPL and complete curves."""

import hashlib
from pathlib import Path

from base.analysis_artifact_paths import sanitize_path_component
from base.analysis_csv_exporter import build_curve_csv_table
from base.csv_writer import write_csv_atomic

from .results import plain_data


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
    payload = plain_data(result["display_payload"])
    x_name, y_name = {
        "SPL": ("时间(s)", "声压级"),
        "FFT": ("频率(Hz)", "幅值"),
        "FBA": ("中心频率(Hz)", "频段声压级"),
    }[instance.analysis_type]
    unit = _csv_unit(payload["y_unit"])
    _, _, rows = build_curve_csv_table(
        instance.config_key, x_name, f"{y_name}{unit}",
        [{"raw_channel": instance.channel_index, "csv_curve": payload}], {},
    )
    header = [x_name, f"{y_name}{unit}", f"下限{unit}", f"上限{unit}"]
    path = _csv_path(task, instance, "曲线")
    write_csv_atomic(path, header, rows)
    return str(path)
