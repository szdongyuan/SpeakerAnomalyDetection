import json
from types import SimpleNamespace
from unittest import mock

import pytest

from ui.operation_sequence import AnalysisModelSelect, OptionList
from base.test_queue_catalog import load_queue_info


def legacy_analysis(valid=True):
    result = {"display_sequence": ["export"], "export": {"type": "Excel", "analysis_channels": [99]}}
    if valid:
        result["display_sequence"].append("spl")
        result["spl"] = {"type": "SPL", "analysis_channels": [0], "limit_checked": True}
    return result


@pytest.mark.parametrize("valid", [False, True])
def test_retired_excel_is_filtered_from_editor_and_saved_payload(ui_qapp, tmp_path, valid):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps([{"seq1": {"acq": {"name": "录制音频", "mode": "RECORD_ONLY", "detail": {}},
                                        "analysis_list": legacy_analysis(valid)}}]), encoding="utf-8")
    original = path.read_bytes()
    host = SimpleNamespace(config=[], mic=None, default_logger=mock.Mock())
    OptionList.init_config_info(host, str(path))
    expected = ["spl"] if valid else []
    assert host.config[0].display_sequence == expected
    assert "export" not in host.config[0].analysis_list
    # Save filtering also handles legacy items supplied directly in memory.
    host.config[0].analysis_list["export"] = {"type": "Excel"}
    host.config[0].display_sequence.append("export")
    editor = SimpleNamespace(auto_analysis_box=SimpleNamespace(isChecked=lambda: True))
    payload = AnalysisModelSelect.format_config_data(editor, host.config)
    saved_analysis = payload[0]["seq1"]["analysis_list"]
    assert saved_analysis["display_sequence"] == expected
    assert "export" not in saved_analysis
    assert path.read_bytes() == original


@pytest.mark.parametrize("valid", [False, True])
def test_retired_excel_is_excluded_from_queue_catalog(tmp_path, valid):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps([{"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000, "total_time": 1}},
                                        "analysis_list": legacy_analysis(valid)}}]), encoding="utf-8")
    original = path.read_bytes()
    info = load_queue_info(str(path))
    assert info["available"] is True
    assert info["analysis_items"] == (["spl"] if valid else [])
    if valid:
        assert info["can_auto_judge"]
    else:
        assert info["can_auto_judge"] is False
        assert info["judgment_reason"] == "仅录音，未配置分析项"
    assert path.read_bytes() == original
