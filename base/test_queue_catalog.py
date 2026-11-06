"""Existing queue catalog and summaries shared by configuration consumers."""
import os
import math
from base.load_config import LoadUiConfig
from base.analysis_config_validation import validate_analysis_config
from base.recording_preview_config import resolve_recording_preview_time_mode
from consts import error_code
from consts.product_test_project_consts import LIMIT_RULE_ANALYSIS_TYPES

def resolve_queue_path(registered_path, registry_dir):
    if not os.path.isabs(registered_path):
        return os.path.abspath(os.path.join(registry_dir, registered_path))
    absolute_path = os.path.abspath(registered_path)
    if os.path.isfile(absolute_path):
        return absolute_path
    file_name = os.path.basename(absolute_path)
    for candidate in (
        os.path.join(registry_dir, file_name),
        os.path.join(os.path.dirname(registry_dir), file_name),
    ):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return absolute_path

def load_queue_info(file_path):
    load_code, queue_data = LoadUiConfig.load_data_from_json(file_path)
    return queue_info_from_data(
        file_path, queue_data if load_code == error_code.OK else None
    )

def queue_info_from_data(file_path, queue_data):
    info = {
        "path": file_path,
        "available": False,
        "acquisition_mode": "",
        "duration": None,
        "analysis_items": [],
        "reason": "",
        "can_auto_judge": False,
        "judgment_reason": "",
    }
    if not isinstance(queue_data, list) or not queue_data:
        info["reason"] = "测试队列文件无法读取"
        return info
    first_sequence_group = queue_data[0]
    if not isinstance(first_sequence_group, dict) or not first_sequence_group:
        info["reason"] = "测试队列没有可执行序列"
        return info
    sequence_data = next(iter(first_sequence_group.values()))
    if not isinstance(sequence_data, dict):
        info["reason"] = "测试队列序列格式错误"
        return info

    info["data"] = queue_data
    acquisition = sequence_data.get("acq", {})
    if not isinstance(acquisition, dict):
        info["reason"] = "测试队列采集配置格式错误"
        return info
    acquisition_mode = str(acquisition.get("mode") or "RECORD_ONLY").strip().upper()
    info["acquisition_mode"] = acquisition_mode
    acquisition_detail = acquisition.get("detail", {})
    analysis_list = sequence_data.get("analysis_list", {})
    if not isinstance(analysis_list, dict):
        info["reason"] = "测试队列分析配置格式错误"
        return info
    info["analysis_list"] = analysis_list
    try:
        validate_analysis_config(analysis_list)
    except ValueError as exc:
        info["reason"] = str(exc)
        return info
    if not isinstance(acquisition_detail, dict):
        info["reason"] = "测试队列采集参数必须是对象"
        return info
    display_sequence = analysis_list.get("display_sequence", [])
    if not isinstance(display_sequence, list) or not all(isinstance(name, str) for name in display_sequence):
        info["reason"] = "分析项顺序必须是名称列表"
        return info
    if acquisition_mode != "RECORD_ONLY":
        info["reason"] = f"不支持的采集模式：{acquisition_mode or '-'}"
        return info
    try:
        resolve_recording_preview_time_mode(acquisition_detail)
    except ValueError as exc:
        info["reason"] = str(exc)
        return info
    duration = acquisition_detail.get("total_time")
    sample_rate = acquisition_detail.get("sample_rate")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        info["reason"] = "录音时长无效"
        return info
    info["duration"] = duration
    if type(sample_rate) not in (int, float) or not math.isfinite(sample_rate) or sample_rate <= 0:
        info["reason"] = "采样率无效"
        return info
    if isinstance(display_sequence, list):
        display_sequence = [
            name for name in display_sequence
            if not (
                isinstance(analysis_list.get(name), dict)
                and analysis_list[name].get("type") == "Excel"
            )
        ]
    info["analysis_items"] = [str(item) for item in display_sequence]
    for item_name in display_sequence:
        if not isinstance(item_name, str) or not isinstance(analysis_list.get(item_name), dict):
            info["reason"] = f"分析项不存在：{item_name}"
            return info
    info["available"] = True
    if not display_sequence:
        info["judgment_reason"] = "仅录音，未配置分析项"
        return info
    if has_automatic_judgment(
        analysis_list,
        display_sequence,
    ):
        info["can_auto_judge"] = True
    else:
        info["judgment_reason"] = "未配置可输出 OK/NG 的规则阈值"
    return info

def has_automatic_judgment(analysis_list, display_sequence):
    for item_name in display_sequence:
        item_config = analysis_list.get(item_name, {})
        normalized_type = str(item_config.get("type", "") or "").upper()
        if normalized_type == "RSC":
            has_reference = bool(
                str(item_config.get("reference_source_path", "") or "").strip()
            )
            current_only = bool(
                item_config.get("view_current_only_without_reference", False)
            )
            threshold_enabled = bool(
                item_config.get("enable_threshold_judgment", True)
            )
            if has_reference and not current_only and threshold_enabled:
                return True
        elif (
            normalized_type in LIMIT_RULE_ANALYSIS_TYPES
            and item_config.get("limit_checked")
        ):
            return True
    return False
