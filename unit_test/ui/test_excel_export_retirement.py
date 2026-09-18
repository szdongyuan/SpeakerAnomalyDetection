import copy
import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget

from main_window import MainWindow
from ui.operation_sequence import AnalysisModelSelect, OptionList
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
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


class CsvHost(SequenceWidgetStreamingOpsMixin, QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self._raw_audio_csv_export_lock = threading.Lock()
        self._raw_audio_csv_export_threads = set()
        self._cleanup_streaming_resources = mock.Mock()
        self.hw_manager = SimpleNamespace(stop=mock.Mock())
        self._shutdown_product_pdf_exporter = mock.Mock()
        # Isolate the old close wrapper: it must not be needed for CSV waiting.
        self.flush_excel_spool_build = lambda **kwargs: []


@pytest.mark.parametrize("boundary", ["main", "visible", "hidden"])
def test_close_waits_for_pending_csv_before_final_cleanup(ui_qapp, tmp_path, boundary):
    host = CsvHost()
    released = threading.Event()
    entered = threading.Event()
    completed = threading.Event()
    csv_path = tmp_path / "raw.csv"

    def write_csv():
        entered.set()
        released.wait()
        csv_path.write_text("sample,value\n0,1\n", encoding="utf-8")
        with host._raw_audio_csv_export_lock:
            host._raw_audio_csv_export_threads.discard(threading.current_thread())
        completed.set()

    class PendingWriter(threading.Thread):
        def join(self, timeout=None):
            released.set()
            super().join(timeout)

    writer = PendingWriter(target=write_csv)
    host._raw_audio_csv_export_threads.add(writer)
    writer.start()
    observations = []
    host.hw_manager.stop.side_effect = lambda: observations.append(("hardware", completed.is_set()))
    host._shutdown_product_pdf_exporter.side_effect = lambda: observations.append(("pdf", completed.is_set()))
    window = None
    try:
        assert entered.wait(2)
        event = QCloseEvent()
        if boundary == "main":
            class Window(MainWindow):
                def __init__(self):
                    QMainWindow.__init__(self)
                    self.sequence_window = host
                    self._close_all_subwindows = mock.Mock()
            window = Window()
            MainWindow.closeEvent(window, event)
        else:
            if boundary == "visible":
                host.show()
            host.closeEvent(event)
        assert event.isAccepted()
        if boundary == "hidden":
            assert not completed.is_set()
            assert observations == []
        else:
            assert completed.is_set()
            assert csv_path.read_text(encoding="utf-8") == "sample,value\n0,1\n"
            assert observations == ([("pdf", True)] if boundary == "main" else [("hardware", True), ("pdf", True)])
    finally:
        released.set()
        writer.join(2)
        host.hide()
        host.deleteLater()
        if window is not None:
            window.deleteLater()
