"""Compatibility checks at the test-queue boundary."""


def validate_analysis_config(analysis_config):
    for name, item in analysis_config.items():
        if isinstance(item, dict) and str(item.get("type", "")).strip().upper() == "AI":
            raise ValueError(
                f"分析项“{name}”使用已移除的 AI 功能，请移除该项或替换为规则分析后再测试。"
            )


def validate_sequence_config(sequence_config):
    for group in sequence_config:
        for sequence in group.values():
            validate_analysis_config(sequence.get("analysis_list", {}))
