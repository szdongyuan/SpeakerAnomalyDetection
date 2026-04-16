import os

from base.load_config import load_config
from base.pre_processing.data_alignment import DataAlignment
from base.training_model_management import TrainingModelManagement
from consts import error_code


def should_validate_model_duration(mode, acq_mode=None):
    # Keep live analysis, imported audio, and history-view analysis consistent.
    # Otherwise the same recording may get an OK/NG result during testing, but
    # later fail when the user clicks "查看".
    return True


def _load_preprocess_config(config_path):
    normalized_path = str(config_path or "").strip()
    if not normalized_path or not os.path.exists(normalized_path):
        return {}
    try:
        result = load_config(config_path=normalized_path, module_name="preprocess")
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def resolve_effective_signal_length(signal_length: int, preprocess_config) -> int:
    total = max(0, int(signal_length))
    if not isinstance(preprocess_config, dict):
        return total

    method = str(preprocess_config.get("preprocess_method") or "").strip().lower()
    if not method or method == "none":
        return total

    process_kwargs = preprocess_config.get("preprocess_param", {}) or {}
    if method == "data_padding":
        return DataAlignment.resolve_padded_signal_length(total, **process_kwargs)

    if method == "sequence_process":
        current = total
        for processor_kwargs in process_kwargs.get("processor_list", []) or []:
            current = resolve_effective_signal_length(current, processor_kwargs)
        return current

    if method == "stack_process":
        resolved_lengths = []
        for processor_kwargs in process_kwargs.get("processor_list", []) or []:
            resolved_lengths.append(resolve_effective_signal_length(total, processor_kwargs))
        unique_lengths = {int(item) for item in resolved_lengths if item is not None}
        if len(unique_lengths) == 1:
            return unique_lengths.pop()
        return total

    return total


def _build_duration_mismatch_message(expected_length: int, raw_length: int, effective_length: int, sample_rate=None):
    current_sample_rate = float(sample_rate or 0)
    if current_sample_rate > 0:
        expected_seconds = expected_length / current_sample_rate
        raw_seconds = raw_length / current_sample_rate
        effective_seconds = effective_length / current_sample_rate
        if effective_length != raw_length:
            return (
                "模型时长不匹配\n"
                f"模型要求: {expected_length} 点 ({expected_seconds:.2f}s)\n"
                f"当前音频: {raw_length} 点 ({raw_seconds:.2f}s)\n"
                f"按预处理后有效长度: {effective_length} 点 ({effective_seconds:.2f}s)"
            )
        return (
            "模型时长不匹配\n"
            f"模型要求: {expected_length} 点 ({expected_seconds:.2f}s)\n"
            f"当前音频: {effective_length} 点 ({effective_seconds:.2f}s)"
        )

    if effective_length != raw_length:
        return (
            "模型时长不匹配\n"
            f"模型要求: {expected_length} 点\n"
            f"当前音频: {raw_length} 点\n"
            f"按预处理后有效长度: {effective_length} 点"
        )
    return (
        "模型时长不匹配\n"
        f"模型要求: {expected_length} 点\n"
        f"当前音频: {effective_length} 点"
    )


def validate_model_duration(model_name, signal_length: int, sample_rate=None, model_manager=None, config_path=None):
    manager = model_manager or TrainingModelManagement()
    query_code, query_result = manager.get_input_dim_info_by_name(model_name)
    if query_code != error_code.OK:
        return False, "模型时长查询失败"

    expected_text = str(query_result).split("x")[0].strip()
    try:
        expected_length = int(float(expected_text))
    except (TypeError, ValueError):
        return False, f"模型输入长度格式无效: {query_result}"

    raw_length = max(0, int(signal_length))
    preprocess_config = _load_preprocess_config(config_path)
    effective_length = resolve_effective_signal_length(raw_length, preprocess_config)
    if expected_length == effective_length:
        return True, None
    return False, _build_duration_mismatch_message(
        expected_length,
        raw_length,
        effective_length,
        sample_rate=sample_rate,
    )


def build_blocked_ai_export_detail(model_name, reason: str, message: str) -> dict:
    return {
        "label": None,
        "blocked_reason": str(reason or "").strip(),
        "blocked_message": str(message or "").strip(),
        "model_name": model_name,
    }
