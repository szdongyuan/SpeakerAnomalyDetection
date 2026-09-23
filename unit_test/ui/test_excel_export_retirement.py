import copy
import json
from types import SimpleNamespace
from unittest import mock

import pytest
from unit_test.ui.test_raw_audio_csv_shutdown import runtime
from PyQt5.QtGui import QCloseEvent

from ui.operation_sequence import AnalysisModelSelect, OptionList
from ui.sequence.analysis_task_builder import AnalysisTaskBuildError, build_analysis_task_request
from unit_test.ui.test_analysis_task_builder import _write_wav
from unit_test.test_product_test_project_config import make_manager


def legacy_analysis(valid=True):
    result = {"display_sequence": ["export"], "export": {"type": "Excel", "analysis_channels": [99]}}
    if valid:
        result["display_sequence"].append("spl")
        result["spl"] = {"type": "SPL", "analysis_channels": [0], "limit_checked": True}
    return result


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_retired_excel_task_entries_are_inert(tmp_path, segmented, valid):
    path = tmp_path / "audio.wav"
    _write_wav(path, (0,))
    analysis = legacy_analysis(valid)
    original = copy.deepcopy(analysis)
    kwargs = dict(condition_key="condition", wav_path=path, source="手动查看",
                  sequence_config=[{"seq1": {"acq": {"detail": {"total_time": 1}}}}],
                  analysis_config=analysis,
                  condition_config={"segmented_analysis": {"mode": "time", "interval_seconds": 1,
                                                            "analysis_seconds": .005}} if segmented else {})
    if valid:
        task = build_analysis_task_request(**kwargs)
        assert [(item.config_key, item.analysis_type) for item in task.instances] == [("spl", "SPL")]
    else:
        with pytest.raises(AnalysisTaskBuildError, match="没有可执行"):
            build_analysis_task_request(**kwargs)
    assert analysis == original


def test_other_unsupported_segmented_items_remain_visible(tmp_path):
    path = tmp_path / "audio.wav"
    _write_wav(path, (0,))
    task = build_analysis_task_request(condition_key="condition", wav_path=path, source="手动查看",
        sequence_config=[{"seq1": {"acq": {"detail": {"total_time": 1}}}}],
        analysis_config={"display_sequence": ["future"], "future": {"type": "Future", "analysis_channels": [0]}},
        condition_config={"segmented_analysis": {"mode": "time", "interval_seconds": 1, "analysis_seconds": .005}})
    assert task.instances[0].analysis_type == "Unsupported"
    assert task.instances[0].parameters["unsupported_type"] == "Future"


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
    manager = make_manager(tmp_path)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps([{"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000, "total_time": 1}},
                                        "analysis_list": legacy_analysis(valid)}}]), encoding="utf-8")
    original = path.read_bytes()
    info = manager._load_queue_info(str(path))
    assert info["available"] is valid
    assert info["analysis_items"] == (["spl"] if valid else [])
    if valid:
        assert info["can_auto_judge"]
    else:
        assert info["reason"] == "未配置分析项"
    assert path.read_bytes() == original


@pytest.mark.parametrize("boundary", ["main", "visible", "hidden"])
def test_close_waits_for_pending_csv_before_final_cleanup(ui_qapp, runtime, tmp_path, boundary):
    # CSV lifecycle lives independently of the retired Excel spool wrapper.
    from PyQt5 import sip
    from unit_test.ui.test_raw_audio_csv_shutdown import Sequence, Window, submit, pump
    service, bridge, gate = runtime
    host = Sequence(bridge)
    host.flush_excel_spool_build = mock.Mock(side_effect=AssertionError("retired Excel spool"))
    window = Window(host, bridge) if boundary == "main" else host
    if boundary != "hidden":
        window.show()
    submit(host, tmp_path)
    try:
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted() is (boundary == "hidden")
        host._shutdown_product_pdf_exporter.assert_not_called()
        gate.set()
        if boundary != "hidden":
            pump(ui_qapp, lambda: not window.isVisible())
            host._shutdown_product_pdf_exporter.assert_called_once()
        else:
            assert service.snapshot().phase == "open"
        host.flush_excel_spool_build.assert_not_called()
    finally:
        sip.delete(window)
        if window is not host:
            sip.delete(host)
