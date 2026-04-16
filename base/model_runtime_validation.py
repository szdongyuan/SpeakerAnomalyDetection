from base.training_model_management import TrainingModelManagement
from consts import error_code


def should_validate_model_duration(mode, acq_mode=None):
    return acq_mode in ["IMPORT_AUDIO"] or str(mode or "").strip().lower() == "view"


def validate_model_duration(model_name, signal_length: int, sample_rate=None, model_manager=None):
    manager = model_manager or TrainingModelManagement()
    query_code, query_result = manager.get_input_dim_info_by_name(model_name)
    if query_code != error_code.OK:
        return False, "模型时长查询失败"

    expected_text = str(query_result).split("x")[0].strip()
    try:
        expected_length = int(float(expected_text))
    except (TypeError, ValueError):
        return False, f"模型输入长度格式无效: {query_result}"

    actual_length = int(signal_length)
    if expected_length == actual_length:
        return True, None

    current_sample_rate = float(sample_rate or 0)
    if current_sample_rate > 0:
        expected_seconds = expected_length / current_sample_rate
        actual_seconds = actual_length / current_sample_rate
        message = (
            "模型时长不匹配\n"
            f"模型要求: {expected_length} 点 ({expected_seconds:.2f}s)\n"
            f"当前音频: {actual_length} 点 ({actual_seconds:.2f}s)"
        )
    else:
        message = (
            "模型时长不匹配\n"
            f"模型要求: {expected_length} 点\n"
            f"当前音频: {actual_length} 点"
        )
    return False, message
