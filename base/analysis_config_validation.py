"""AI drafts may be saved, but cannot enter an unimplemented inference path."""

from base.ai_model_catalog import ai_execution_issue


def validate_analysis_config(analysis_config, *, for_execution=False):
    for name, item in analysis_config.items():
        if isinstance(item, dict) and str(item.get("type", "")).strip().upper() == "AI":
            if item["type"] != "AI":
                raise ValueError(f"分析项“{name}”的 AI 类型必须写为 AI。")
            model_name = item.get("analyse_model_name", "")
            if not isinstance(model_name, str):
                raise ValueError(f"分析项“{name}”的模型名称必须为字符串。")
            if for_execution:
                _, message = ai_execution_issue(item)
                raise ValueError(f"分析项“{name}”：{message}")


def validate_sequence_config(sequence_config):
    for group in sequence_config:
        for sequence in group.values():
            validate_analysis_config(sequence.get("analysis_list", {}))
