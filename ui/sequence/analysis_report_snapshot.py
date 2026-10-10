from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any

def build_segment_report_results(task_result, analysis_config):
    results = []
    for segment in task_result.segments:
        plan = segment.segment
        result = replace(task_result, instance_results=segment.instance_results, segments=())
        results.append({
            **asdict(plan), "segment_value": plan.value, "segment_label": plan.label,
            "window_start_seconds": plan.window_start_sample / plan.sample_rate,
            "window_end_seconds": plan.window_end_sample / plan.sample_rate,
            "result": segment.final_judgement or ("未产生判定" if segment.execution_status == "分析完成" else "分析失败"),
            "analysis_state": "completed" if segment.execution_status == "分析完成" else "failed",
            "analysis_items": build_analysis_report_items_from_task_result(result, analysis_config),
        })
    return results


def build_analysis_report_items_from_task_result(task_result, analysis_config):
    """Build segment analysis items from a headless worker result."""
    config = analysis_config if isinstance(analysis_config, dict) else {}
    counts = {}
    for item in task_result.instance_results:
        counts[item.config_key] = counts.get(item.config_key, 0) + 1

    report_items = []
    for item in task_result.instance_results:
        item_config = config.get(item.config_key, {})
        if not isinstance(item_config, dict):
            item_config = {}
        metrics = item.metrics.to_dict()
        if item.execution_status != "分析完成":
            state = "failed"
            status = "分析失败"
        elif item.contributes_to_final:
            state = "completed"
            status = item.judgement or "未产生判定"
        elif item.analysis_type == "Spec":
            state = "completed"
            status = "仅图表分析"
        else:
            state = "completed"
            status = "未启用判定"
        measurement, unit = _process_result_measurement(item.analysis_type, metrics)
        lower_limit, upper_limit = _process_result_limits(
            item.analysis_type,
            metrics,
            item_config,
        )
        images = []
        image_errors = []
        csv_paths = []
        artifact_errors = []
        for artifact in item.artifacts:
            if artifact.kind == "图片" and artifact.status == "已保存":
                try:
                    png_data = Path(artifact.path).read_bytes()
                except OSError as error:
                    image_errors.append(str(error))
                else:
                    images.append(
                        {
                            "caption": f"{item.config_key} - CH{item.raw_channel + 1}",
                            "png_data": png_data,
                        }
                    )
            elif artifact.kind.startswith("CSV:") and artifact.status == "已保存":
                csv_paths.append(artifact.path)
            elif artifact.status == "保存失败":
                artifact_errors.append(
                    f"{artifact.kind}保存失败："
                    f"{artifact.error_message or '未知错误'}"
                )
        report_name = (
            item.runtime_key if counts.get(item.config_key, 0) > 1 else item.config_key
        )
        report_items.append(
            {
                "name": report_name,
                "item_key": item.config_key,
                "runtime_key": item.runtime_key,
                "channel_key": f"In{item.raw_channel + 1}",
                "channel_label": getattr(task_result, "channel_labels", {}).get(f"CH{item.raw_channel + 1}", ""),
                "type": item.analysis_type,
                "state": state,
                "status": status,
                "deviation": "-",
                "measurement": measurement,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "unit": unit,
                "error": "；".join(
                    value
                    for value in (item.error_message, *artifact_errors)
                    if value
                ),
                "image_errors": image_errors,
                "images": images,
                "csv_summary": "；".join(csv_paths),
            }
        )
    return report_items


def _process_result_measurement(analysis_type, metrics):
    if analysis_type == "SPL":
        return _format_measurement(metrics.get("overall_spl")), str(
            metrics.get("unit") or "dB"
        )
    if analysis_type == "FBA":
        weighting = str(metrics.get("weighting") or "Z")
        unit = "dB" if weighting == "Z" else f"dB({weighting})"
        return _format_measurement(metrics.get("overall_weighted_db")), unit
    if analysis_type == "FFT":
        weighting = str(metrics.get("weighting") or "Z")
        unit = (
            "dB"
            if metrics.get("display_mode") == "delta"
            else ("dB SPL" if weighting == "Z" else f"dB({weighting}) SPL")
        )
        return _format_measurement(metrics.get("peak_value")), unit
    if analysis_type == "Spec":
        return "-", "dB"
    return "-", "-"


def _process_result_limits(analysis_type, metrics, config):
    if analysis_type == "SPL" and str(
        config.get("limit_metric", "curve_y") or "curve_y"
    ).lower() == "overall_spl":
        return (
            _format_measurement(metrics.get("overall_lower_limit")),
            _format_measurement(metrics.get("overall_upper_limit")),
        )
    if not config.get("limit_checked", False):
        return "-", "-"
    return (
        metrics.get("lower_limit_summary") or _configured_limit_value(config, "lower", {}),
        metrics.get("upper_limit_summary") or _configured_limit_value(config, "upper", {}),
    )


def _finite_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_finite_values(item))
        return values
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if math.isfinite(number) else []


def _format_measurement(value: Any) -> str:
    values = _finite_values(value)
    if not values:
        return "-"
    return f"{values[0]:.6g}"


def _format_limit_range(value: Any) -> str:
    values = _finite_values(value)
    if not values:
        return "-"
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-12):
        return f"{lower:.6g}"
    return f"{lower:.6g} ~ {upper:.6g}"


def _configured_limit_value(
    item_config: dict[str, Any],
    side: str,
    result: dict[str, Any],
    *,
    preferred_prefixes: tuple[str, ...] | None = None,
) -> str:
    if (
        str(item_config.get("type") or "").upper() == "SPL"
        and preferred_prefixes == ("scalar",)
    ):
        if not item_config.get("limit_checked", False) or not item_config.get(
            f"scalar_{side}_enabled", side == "upper"
        ):
            return "-"
        return _format_measurement(
            item_config.get(f"scalar_{side}_value", 100.0 if side == "upper" else 0.0)
        )
    result_key = f"{side}_limits"
    if result_key in result:
        return _format_limit_range(result.get(result_key))
    if not item_config.get("limit_checked", False):
        return "-"

    for prefix in preferred_prefixes or ("constant", "scalar", "curve"):
        enabled_key = f"{prefix}_{side}_enabled"
        value_key = f"{prefix}_{side}_value"
        if item_config.get(enabled_key, False):
            return _format_measurement(item_config.get(value_key))

    limit_data = item_config.get("limit_data")
    if isinstance(limit_data, (list, tuple)) and len(limit_data) >= 3:
        limit_index = 1 if side == "upper" else 2
        return _format_limit_range(limit_data[limit_index])

    segment_keys = (
        f"manual_{side}_segments",
        f"{side}_segments",
    )
    if any(item_config.get(key) for key in segment_keys):
        return "曲线"
    if str(item_config.get("limit_mode") or "").lower() in ("csv", "manual"):
        return "曲线"
    return "-"
