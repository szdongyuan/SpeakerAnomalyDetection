import pytest

from base.analysis_config_validation import ai_execution_issue


@pytest.mark.parametrize("config, kind", [
    ({}, "MissingAiModel"),
    ({"analyse_model_name": "candidate"}, "AiInferenceUnavailable"),
])
def test_runtime_is_explicitly_blocked(config, kind):
    assert ai_execution_issue(config)[0] == kind
