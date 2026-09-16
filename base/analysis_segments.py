"""Validated configuration and deterministic recording-relative segment windows."""

from dataclasses import dataclass
from decimal import Decimal
import math

from base.config_number_format import format_config_number

TIME_UNIT_SECONDS = {"s": 1, "min": 60, "h": 3600}


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}必须是大于零的有限数值")
    return float(value)


def normalize_segmented_analysis(condition):
    settings = condition.get("segmented_analysis")
    if settings is None:
        legacy = condition.get("output_load")
        if legacy is None:
            return {"mode": "none"}
        if not isinstance(legacy, dict) or type(legacy.get("enabled")) is not bool:
            raise ValueError("输出负载配置及启用状态无效")
        settings = {
            "mode": "output_load" if legacy["enabled"] else "none",
            "load_values": legacy.get("values", []),
            "load_unit": "A",
            "analysis_seconds": legacy.get("analysis_seconds", 10),
        }
    if not isinstance(settings, dict):
        raise ValueError("分段分析配置必须是对象")
    mode = settings.get("mode", "none")
    if mode not in {"none", "output_load", "time"}:
        raise ValueError("分段方式必须是不分段、按输出负载或按时间")
    result = {"mode": mode}
    if mode == "none":
        return result
    result["analysis_seconds"] = _positive(settings.get("analysis_seconds"), "每段分析时长")
    if mode == "output_load":
        values = settings.get("load_values")
        if not isinstance(values, list) or not values:
            raise ValueError("请至少输入一个输出负载")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("输出负载必须是有限的非负数列表")
        unit = settings.get("load_unit", "A")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("请选择或输入负载单位")
        unit = unit.strip()
        result.update(load_values=list(values), load_unit=unit)
    else:
        result["interval_seconds"] = _positive(settings.get("interval_seconds"), "分段间隔")
        unit = settings.get("display_time_unit", "s")
        if unit not in TIME_UNIT_SECONDS:
            raise ValueError("时间单位必须为秒、分钟或小时")
        result["display_time_unit"] = unit
    return result


def segment_condition_fields(condition):
    """Normalize once at configuration boundaries; do not dual-write legacy keys."""
    result = {}
    if "input_voltage" in condition:
        voltage = condition["input_voltage"]
        if not isinstance(voltage, str):
            raise ValueError("输入电压必须是文字，可留空")
        result["input_voltage"] = voltage.strip()
    if "segmented_analysis" in condition or "output_load" in condition:
        result["segmented_analysis"] = normalize_segmented_analysis(condition)
    return result


def segment_count(settings, total_seconds):
    if settings["mode"] == "none":
        return 0
    total = _positive(total_seconds, "测试队列录音时长")
    if settings["mode"] == "output_load":
        count = len(settings["load_values"])
    else:
        ratio = total / settings["interval_seconds"]
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8) or round(ratio) < 1:
            raise ValueError("录音时长必须是分段间隔的整数倍，请调整分段间隔")
        count = round(ratio)
    # Compare saved decimal values without introducing division rounding at equality.
    if Decimal(str(settings["analysis_seconds"])) * count > Decimal(str(total)):
        raise ValueError("每段分析时长不能超过分段时长")
    return count


@dataclass(frozen=True)
class AnalysisSegment:
    segment_index: int
    label: str
    mode: str
    value: float
    unit: str
    start_sample: int
    end_sample: int
    window_start_sample: int
    window_end_sample: int
    sample_rate: int

    def available(self, frame_count):
        return self.window_end_sample <= frame_count


def build_segment_plan(settings, total_seconds, sample_rate):
    settings = normalize_segmented_analysis({"segmented_analysis": settings})
    count = segment_count(settings, total_seconds)
    if not count:
        return ()
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("采样率必须是正整数")
    frames = round(total_seconds * sample_rate)
    window = round(settings["analysis_seconds"] * sample_rate)
    if window < 1:
        raise ValueError("每段分析时长不足一个采样点")
    boundaries = [round(i * frames / count) for i in range(count + 1)]
    segments = []
    values = settings.get("load_values", [])
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if window > end - start:
            raise ValueError("每段分析时长超过实际分段采样点数")
        if settings["mode"] == "output_load":
            value, unit = values[index], settings["load_unit"]
            label = f"输出负载{format_config_number(value)}{unit}"
            if values.count(value) > 1:
                label += f"_第{index + 1}段"
        else:
            unit = settings["display_time_unit"]
            value = float(Decimal(str(settings["interval_seconds"])) * (index + 1)
                          / TIME_UNIT_SECONDS[unit])
            label = f"时间{format_config_number(value)}{unit}"
        window_start = start + (end - start - window) // 2
        segments.append(AnalysisSegment(index, label, settings["mode"], value, unit,
                                        start, end, window_start, window_start + window, sample_rate))
    return tuple(segments)


def recording_duration(sequence_config):
    """Resolve the frozen queue duration, never substitute a short WAV length."""
    for entry in sequence_config or []:
        for sequence in entry.values():
            acq = sequence.get("acq", {})
            duration = acq.get("detail", {}).get("total_time")
            if duration is not None:
                return _positive(duration, "测试队列录音时长")
    raise ValueError("分段分析缺少录音时长快照，请先配置测试队列录音时长")
