import numpy as np
import pytest
import soundfile as sf

from base.analysis.input_adapter import build_analysis_batch
from base.analysis.executor import execute_task


@pytest.mark.parametrize("name, kind", [("", "MissingAiModel"), ("future", "AiInferenceUnavailable")])
def test_ai_failure_never_produces_a_judgement(batch_payload, name, kind):
    batch_payload["tasks"] = batch_payload["tasks"][:1]
    raw = batch_payload["tasks"][0]
    raw["analysis_config"] = {
        "display_sequence": ["AI 1"],
        "AI 1": {"type": "AI", "analysis_channel": 0, "analyse_model_name": name},
    }
    sf.write(raw["wav_path"], np.zeros((100, 4)), 48000)
    batch = build_analysis_batch(batch_payload)
    assert any(issue.error_type == kind for issue in batch.preparation_issues)
    task = batch.tasks[0]
    result = execute_task(task)
    item = result["instance_results"][0]
    assert result["execution_status"] == "分析失败"
    assert item["error"]["type"] == kind
    assert item["judgement"] is None
    assert item["judgement_status"] == "not_executed"
    assert item["metrics"] == {}
