import threading
import time
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QWidget

from base.product_test_pdf_report import ProductPdfExportResult
from ui.sequence.sequence_widget_product_pdf_ops import (
    SequenceWidgetProductPdfOpsMixin,
)


class _PdfRuntimeHost(SequenceWidgetProductPdfOpsMixin):
    def __init__(self):
        self.product_test_pdf_report_config = {
            "enabled": True,
            "save_dir": "D:/reports",
        }
        self.product_test_condition_configs = [
            {"key": "01", "condition_name": "6000 rpm"},
            {"key": "02", "condition_name": "7000 rpm"},
        ]
        self.recent_test_session_by_id = {
            "session-01": {
                "session_id": "session-01",
                "group_id": "group-1",
                "condition_key": "01",
                "created_at": "2026-08-06T10:30:20",
                "time_text": "2026-08-06 10:30:20",
                "recorded_path": "D:/audio/6000.wav",
                "sample_rate": 48000,
                "recorded_signal_info": {"labels": "OK"},
                "analysis_result_dict": {
                    "声压级": (True, 0.2),
                },
                "analysis_report_state": "completed",
                "analysis_report_items": [
                    {
                        "name": "声压级",
                        "type": "SPL",
                        "state": "completed",
                        "status": "OK",
                        "deviation": "0.2",
                        "images": [],
                    }
                ],
            },
            "session-02": {
                "session_id": "session-02",
                "group_id": "group-1",
                "condition_key": "02",
                "created_at": "2026-08-06T10:30:25",
                "time_text": "2026-08-06 10:30:25",
                "recorded_path": "D:/audio/7000.wav",
                "sample_rate": 48000,
                "recorded_signal_info": {"labels": "OK"},
                "analysis_result_dict": {
                    "频段能量": (True, 0.1),
                },
                "analysis_report_state": "completed",
                "analysis_report_items": [
                    {
                        "name": "频段能量",
                        "type": "FBA",
                        "state": "completed",
                        "status": "OK",
                        "deviation": "0.1",
                        "images": [],
                    }
                ],
            },
        }
        self.recent_session_panel = SimpleNamespace(
            group_records={
                "group-1": {
                    "group_id": "group-1",
                    "barcode": "SN123456",
                    "product_model": "S004-1",
                    "time_text": "2026-08-06 10:30:20",
                    "session_ids": {
                        "01": "session-01",
                        "02": "session-02",
                    },
                    "results": {
                        "01": "OK",
                        "02": "OK",
                    },
                }
            }
        )
        self._condition_record_cache = {}
        self._product_pdf_report_states = {}
        self._product_pdf_report_paths = {}

    def _product_condition_sequence(self):
        return [dict(item) for item in self.product_test_condition_configs]

    @staticmethod
    def _product_condition_runtime_key(condition, index=0):
        return str(condition.get("key") or index)

    @staticmethod
    def _normalize_recent_session_storage_label(value):
        normalized = str(value or "").strip()
        return normalized if normalized in ("OK", "NG", "not_labeled") else ""


class _AsyncPdfRuntimeHost(_PdfRuntimeHost, QWidget):
    def __init__(self):
        QWidget.__init__(self)
        _PdfRuntimeHost.__init__(self)


def test_runtime_exports_one_pdf_for_complete_product(monkeypatch):
    host = _PdfRuntimeHost()
    calls = []

    def fake_export(config, report_data):
        calls.append((config, report_data))
        return ProductPdfExportResult(
            ok=True,
            message="ok",
            file_path="D:/reports/product.pdf",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    output_path = host._maybe_export_product_test_pdf("group-1", "OK")

    assert output_path == "D:/reports/product.pdf"
    assert len(calls) == 1
    report_data = calls[0][1]
    assert report_data["barcode"] == "SN123456"
    assert report_data["product_model"] == "S004-1"
    assert report_data["overall_result"] == "OK"
    assert [item["name"] for item in report_data["conditions"]] == [
        "6000 rpm",
        "7000 rpm",
    ]
    assert report_data["conditions"][0]["analysis_results"] == [
        {"name": "声压级", "status": "OK", "deviation": "0.2"},
    ]
    assert report_data["conditions"][0]["analysis_items"][0]["type"] == "SPL"


def test_runtime_group_resolves_canonical_session_analysis_images():
    host = _PdfRuntimeHost()
    png_data = b"canonical-png"
    canonical_item = host.recent_test_session_by_id["session-01"][
        "analysis_report_items"
    ][0]
    canonical_item["images"] = [{"png_data": png_data}]

    group = host._product_pdf_group_data("group-1")
    report_data = host._build_product_pdf_report_data("group-1", "OK")

    assert group["records"]["01"] is host.recent_test_session_by_id[
        "session-01"
    ]
    report_item = report_data["conditions"][0]["analysis_items"][0]
    assert report_item is canonical_item
    assert report_item["images"][0]["png_data"] is png_data


def test_runtime_pdf_and_unified_product_results_share_one_summary():
    host = _PdfRuntimeHost()
    host.recent_test_session_by_id["session-02"]["result_label"] = "NG"
    host.recent_test_session_by_id["session-02"]["recorded_signal_info"][
        "labels"
    ] = "NG"

    assert host._product_group_result_state("group-1") == (True, "NG")
    assert host._product_pdf_group_result_state("group-1") == (True, "NG")


def test_runtime_uses_import_condition_cache_without_recent_sessions(monkeypatch):
    host = _PdfRuntimeHost()
    host.recent_test_session_by_id = {}
    host.recent_session_panel.group_records = {}
    host._condition_record_cache = {
        "01": {
            "source_type": "imported",
            "group_id": "group-import",
            "condition_key": "01",
            "created_at": "2026-08-06T10:30:20",
            "time_text": "2026-08-06 10:30:20",
            "product_model": "S004-1",
            "barcode": "",
            "recorded_path": "D:/audio/import-6000.wav",
            "sample_rate": 44100,
            "recorded_signal_info": {"labels": "OK"},
            "result_label": "OK",
            "analysis_result_dict": {"声压级": (True, 0.2)},
            "analysis_report_state": "completed",
            "analysis_report_items": [],
        },
        "02": {
            "source_type": "imported",
            "group_id": "group-import",
            "condition_key": "02",
            "created_at": "2026-08-06T10:30:25",
            "time_text": "2026-08-06 10:30:25",
            "product_model": "S004-1",
            "barcode": "",
            "recorded_path": "D:/audio/import-7000.wav",
            "sample_rate": 44100,
            "recorded_signal_info": {"labels": "NG"},
            "result_label": "NG",
            "analysis_result_dict": {"频段能量": (False, 0.5)},
            "analysis_report_state": "completed",
            "analysis_report_items": [],
        },
    }
    calls = []

    def fake_export(_config, report_data):
        calls.append(report_data)
        return ProductPdfExportResult(
            ok=True,
            message="ok",
            file_path="D:/reports/import-product.pdf",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    output_path = host._try_export_product_test_pdf("group-import")

    assert output_path == "D:/reports/import-product.pdf"
    assert len(calls) == 1
    assert calls[0]["overall_result"] == "NG"
    assert [item["recorded_path"] for item in calls[0]["conditions"]] == [
        "D:/audio/import-6000.wav",
        "D:/audio/import-7000.wav",
    ]


def test_runtime_exports_single_condition_product_without_multi_condition_ui_state(
    monkeypatch,
):
    host = _PdfRuntimeHost()
    host.product_test_condition_configs = [
        {"key": "01", "condition_name": "6000 rpm"},
    ]
    host.recent_session_panel.group_records["group-1"]["session_ids"] = {
        "01": "session-01",
    }
    host.recent_session_panel.group_records["group-1"]["results"] = {
        "01": "OK",
    }
    host.recent_test_session_by_id.pop("session-02")
    calls = []

    def fake_export(_config, report_data):
        calls.append(report_data)
        return ProductPdfExportResult(
            ok=True,
            message="ok",
            file_path="D:/reports/single-product.pdf",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    output_path = host._try_export_product_test_pdf("group-1")

    assert output_path == "D:/reports/single-product.pdf"
    assert len(calls) == 1
    assert calls[0]["overall_result"] == "OK"
    assert [item["key"] for item in calls[0]["conditions"]] == ["01"]


def test_runtime_waits_until_every_condition_analysis_is_terminal(monkeypatch):
    host = _PdfRuntimeHost()
    host.recent_test_session_by_id["session-02"][
        "analysis_report_state"
    ] = "pending"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("工况分析未完成时不应导出 PDF")

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fail_if_called,
    )

    assert host._maybe_export_product_test_pdf("group-1", "OK") is None


def test_runtime_treats_failed_analysis_as_terminal(monkeypatch):
    host = _PdfRuntimeHost()
    host.recent_test_session_by_id["session-02"][
        "analysis_report_state"
    ] = "failed"
    calls = []

    def fake_export(_config, _report_data):
        calls.append(True)
        return ProductPdfExportResult(True, "ok", "D:/reports/product.pdf")

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    assert host._maybe_export_product_test_pdf("group-1", "NG") == "D:/reports/product.pdf"
    assert calls == [True]


def test_runtime_does_not_export_unchanged_product_twice(monkeypatch):
    host = _PdfRuntimeHost()
    calls = []

    def fake_export(_config, _report_data):
        calls.append(True)
        return ProductPdfExportResult(True, "ok", "D:/reports/product.pdf")

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    host._maybe_export_product_test_pdf("group-1", "OK")
    host._maybe_export_product_test_pdf("group-1", "OK")

    assert len(calls) == 1


def test_runtime_skips_pdf_when_program_setting_is_disabled(monkeypatch):
    host = _PdfRuntimeHost()
    host.product_test_pdf_report_config["enabled"] = False

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("禁用PDF报告时不应调用导出器")

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fail_if_called,
    )

    assert host._maybe_export_product_test_pdf("group-1", "OK") is None


def test_runtime_retries_after_pdf_export_failure(monkeypatch):
    host = _PdfRuntimeHost()
    calls = []
    outcomes = [
        ProductPdfExportResult(False, "保存失败"),
        ProductPdfExportResult(
            True,
            "ok",
            "D:/reports/retried-product.pdf",
        ),
    ]

    def fake_export(_config, _report_data):
        calls.append(True)
        return outcomes.pop(0)

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )

    assert host._maybe_export_product_test_pdf("group-1", "OK") is None
    assert host._product_pdf_report_failures["group-1"] == "保存失败"

    assert (
        host._retry_product_pdf_export("group-1")
        == "D:/reports/retried-product.pdf"
    )
    assert len(calls) == 2
    assert "group-1" not in host._product_pdf_report_failures
    assert host._product_pdf_manual_retry_groups == set()


def test_old_group_pdf_failure_only_logs_during_new_round():
    host = _PdfRuntimeHost()
    stage_calls = []
    dialog_calls = []
    error_logs = []
    host._product_pdf_report_generation = 0
    host._current_manual_product_display_group_id = lambda: "group-2"
    host.left_panel = SimpleNamespace(
        set_current_stage=lambda *args, **kwargs: stage_calls.append(
            (args, kwargs)
        )
    )
    host.default_logger = SimpleNamespace(
        error=lambda message: error_logs.append(message)
    )
    host._show_product_pdf_failure_retry = (
        lambda *args: dialog_calls.append(args)
    )

    host._handle_product_pdf_export_result(
        (0, "group-1"),
        None,
        ProductPdfExportResult(False, "保存失败"),
    )

    assert error_logs == ["保存失败"]
    assert host._product_pdf_report_failures["group-1"] == "保存失败"
    assert stage_calls == []
    assert dialog_calls == []


def test_failure_dialog_shows_reason_and_retries_without_blocking(monkeypatch):
    app = QApplication.instance() or QApplication([])
    host = _AsyncPdfRuntimeHost()
    retry_calls = []

    class _Signal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class _MessageBox:
        Warning = 1
        AcceptRole = 2
        RejectRole = 3

        def __init__(self, parent):
            self.parent = parent
            self.buttonClicked = _Signal()
            self.finished = _Signal()
            self.informative_text = ""
            self.buttons = []
            self.shown = False
            self.closed = False

        def setAttribute(self, *_args):
            pass

        def setWindowModality(self, *_args):
            pass

        def setIcon(self, *_args):
            pass

        def setWindowTitle(self, *_args):
            pass

        def setText(self, *_args):
            pass

        def setInformativeText(self, text):
            self.informative_text = text

        def addButton(self, text, role):
            button = SimpleNamespace(text=text, role=role)
            self.buttons.append(button)
            return button

        def show(self):
            self.shown = True

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.QMessageBox",
        _MessageBox,
    )
    host._retry_product_pdf_export = lambda group_id: retry_calls.append(
        group_id
    )

    host._show_product_pdf_failure_retry(
        "group-1",
        "PDF报告文件被占用",
    )

    dialog = host._product_pdf_failure_dialog
    assert dialog.shown
    assert dialog.informative_text == "PDF报告文件被占用"
    retry_button = next(
        button for button in dialog.buttons if button.text == "重试生成"
    )
    dialog.buttonClicked.callback(retry_button)
    assert dialog.closed
    assert retry_calls == ["group-1"]
    host.close()
    app.processEvents()


def test_manual_retry_success_shows_nonblocking_confirmation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    host = _AsyncPdfRuntimeHost()

    class _Signal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class _MessageBox:
        Information = 1
        AcceptRole = 2

        def __init__(self, parent):
            self.parent = parent
            self.finished = _Signal()
            self.window_title = ""
            self.text = ""
            self.informative_text = ""
            self.buttons = []
            self.shown = False

        def setAttribute(self, *_args):
            pass

        def setWindowModality(self, *_args):
            pass

        def setIcon(self, *_args):
            pass

        def setWindowTitle(self, title):
            self.window_title = title

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def addButton(self, text, role):
            self.buttons.append((text, role))

        def show(self):
            self.shown = True

        def close(self):
            pass

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.QMessageBox",
        _MessageBox,
    )
    host._product_pdf_manual_retry_groups = {"group-1"}

    output_path = host._handle_product_pdf_export_result(
        (0, "group-1"),
        ("signature",),
        ProductPdfExportResult(
            True,
            "ok",
            "D:/reports/retried-product.pdf",
        ),
    )

    assert output_path == "D:/reports/retried-product.pdf"
    assert host._product_pdf_manual_retry_groups == set()
    dialog = host._product_pdf_success_dialog
    assert dialog.shown
    assert dialog.window_title == "PDF报告生成成功"
    assert dialog.text == "PDF 报告已重新生成。"
    assert dialog.informative_text == "D:/reports/retried-product.pdf"
    assert dialog.buttons == [("关闭", _MessageBox.AcceptRole)]
    dialog.finished.callback()
    assert host._product_pdf_success_dialog is None
    assert host._product_pdf_success_dialog_group_id is None
    host.close()
    app.processEvents()


def test_reset_cancels_only_queued_pdf_jobs():
    host = _PdfRuntimeHost()

    class _Future:
        def __init__(self, cancellable):
            self.cancellable = cancellable
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1
            return self.cancellable

    running_key = (0, "running-group")
    queued_key = (0, "queued-group")
    running_future = _Future(False)
    queued_future = _Future(True)
    dialog_closed = []
    success_dialog_closed = []
    host._product_pdf_report_generation = 0
    host._product_pdf_export_jobs = {
        running_key: running_future,
        queued_key: queued_future,
    }
    host._product_pdf_pending_exports = {
        running_key: ("config", "report"),
        queued_key: ("config", "report"),
    }
    host._product_pdf_failure_dialog = SimpleNamespace(
        close=lambda: dialog_closed.append(True)
    )
    host._product_pdf_failure_dialog_group_id = "queued-group"
    host._product_pdf_success_dialog = SimpleNamespace(
        close=lambda: success_dialog_closed.append(True)
    )
    host._product_pdf_success_dialog_group_id = "running-group"
    host._product_pdf_manual_retry_groups = {"running-group"}

    host._reset_product_pdf_report_tracking()

    assert host._product_pdf_report_generation == 1
    assert host._product_pdf_export_jobs == {
        running_key: running_future,
    }
    assert running_future.cancel_calls == 1
    assert queued_future.cancel_calls == 1
    assert host._product_pdf_pending_exports == {}
    assert dialog_closed == [True]
    assert success_dialog_closed == [True]
    assert host._product_pdf_failure_dialog is None
    assert host._product_pdf_failure_dialog_group_id is None
    assert host._product_pdf_success_dialog is None
    assert host._product_pdf_success_dialog_group_id is None
    assert host._product_pdf_manual_retry_groups == set()


def test_qt_runtime_generates_pdf_without_blocking_ui_thread(monkeypatch):
    app = QApplication.instance() or QApplication([])
    host = _AsyncPdfRuntimeHost()
    release_export = threading.Event()

    def fake_export(_config, _report_data):
        release_export.wait(timeout=2.0)
        return ProductPdfExportResult(
            True,
            "ok",
            "D:/reports/background-product.pdf",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )
    release_timer = threading.Timer(0.3, release_export.set)
    release_timer.start()
    try:
        started_at = time.perf_counter()
        output_path = host._maybe_export_product_test_pdf("group-1", "OK")
        elapsed = time.perf_counter() - started_at

        assert output_path is None
        assert elapsed < 0.2
        future = next(iter(host._product_pdf_export_jobs.values()))
        future.result(timeout=2.0)
        host._poll_product_pdf_exports()
        assert (
            host._product_pdf_report_paths["group-1"]
            == "D:/reports/background-product.pdf"
        )
    finally:
        release_export.set()
        release_timer.cancel()
        host._shutdown_product_pdf_exporter()
        host.close()
        app.processEvents()


def test_background_export_coalesces_updates_and_runs_latest_report(monkeypatch):
    app = QApplication.instance() or QApplication([])
    host = _AsyncPdfRuntimeHost()
    release_first_export = threading.Event()
    exported_results = []

    def fake_export(_config, report_data):
        exported_results.append(report_data["overall_result"])
        if len(exported_results) == 1:
            release_first_export.wait(timeout=2.0)
        return ProductPdfExportResult(
            True,
            "ok",
            f"D:/reports/{report_data['overall_result']}.pdf",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_product_pdf_ops.export_product_test_pdf",
        fake_export,
    )
    try:
        host._maybe_export_product_test_pdf("group-1", "OK")
        host._maybe_export_product_test_pdf("group-1", "NG")
        assert len(host._product_pdf_pending_exports) == 1

        release_first_export.set()
        first_future = next(iter(host._product_pdf_export_jobs.values()))
        first_future.result(timeout=2.0)
        host._poll_product_pdf_exports()
        second_future = next(iter(host._product_pdf_export_jobs.values()))
        second_future.result(timeout=2.0)
        host._poll_product_pdf_exports()

        assert exported_results == ["OK", "NG"]
        assert host._product_pdf_report_paths["group-1"] == "D:/reports/NG.pdf"
    finally:
        release_first_export.set()
        host._shutdown_product_pdf_exporter()
        host.close()
        app.processEvents()
