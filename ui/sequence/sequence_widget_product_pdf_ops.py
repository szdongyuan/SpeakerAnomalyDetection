import copy
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import QCoreApplication, QObject, Qt, QTimer
from PyQt5.QtWidgets import QMessageBox, QWidget

from base.product_test_pdf_report import (
    ProductPdfExportResult,
    export_product_test_pdf,
    product_report_signature,
)
from ui.sequence.product_condition_result_ops import (
    SequenceWidgetProductConditionResultOpsMixin,
)


def _execute_product_pdf_export(
    report_config,
    report_data,
    previous_signature=None,
    previous_path=None,
):
    try:
        signature = product_report_signature(report_data)
        if previous_signature == signature:
            return (
                signature,
                ProductPdfExportResult(
                    ok=True,
                    message="PDF报告内容未变化",
                    file_path=previous_path,
                ),
            )
        return signature, export_product_test_pdf(report_config, report_data)
    except Exception as exc:
        return (
            None,
            ProductPdfExportResult(
                ok=False,
                message=f"PDF报告生成异常：{exc}",
            ),
        )


class SequenceWidgetProductPdfOpsMixin(
    SequenceWidgetProductConditionResultOpsMixin
):
    _PRODUCT_PDF_TERMINAL_ANALYSIS_STATES = {
        "completed",
        "failed",
        "not_required",
    }

    @staticmethod
    def _format_product_pdf_deviation(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _product_pdf_analysis_rows(analysis_result_dict):
        rows = []
        if not isinstance(analysis_result_dict, dict):
            return rows
        for name, result in analysis_result_dict.items():
            status = "-"
            deviation = "-"
            if isinstance(result, (tuple, list)) and result:
                if result[0] is not None:
                    status = "OK" if bool(result[0]) else "NG"
                if len(result) > 1:
                    deviation = SequenceWidgetProductPdfOpsMixin._format_product_pdf_deviation(
                        result[1]
                    )
            rows.append(
                {
                    "name": str(name or ""),
                    "status": status,
                    "deviation": deviation,
                }
            )
        return rows

    def _product_pdf_group_data(self, group_id):
        return self._collect_product_condition_records(group_id)

    def _build_product_pdf_report_data(self, group_id, overall_result):
        group = self._product_pdf_group_data(group_id)
        if not isinstance(group, dict):
            return None

        records = group.get("records") or {}
        conditions = []
        first_record = None
        for index, condition in enumerate(self._product_condition_sequence()):
            key = self._product_condition_runtime_key(condition, index)
            record = records.get(key)
            if not isinstance(record, dict):
                continue
            if first_record is None:
                first_record = record
            recorded_signal_info = record.get("recorded_signal_info") or {}
            result_label = recorded_signal_info.get("labels") or record.get("result_label")
            normalize_label = getattr(self, "_normalize_recent_session_storage_label", None)
            if callable(normalize_label):
                result_label = normalize_label(result_label)
            result_text = str(result_label or "-")
            conditions.append(
                {
                    "key": key,
                    "name": str(
                        condition.get("condition_name")
                        or condition.get("name")
                        or key
                    ),
                    "result": result_text,
                    "recorded_path": str(record.get("recorded_path") or ""),
                    "record_time": str(record.get("time_text") or ""),
                    "sample_rate": record.get("sample_rate") or "-",
                    "analysis_state": str(
                        record.get("analysis_report_state") or ""
                    ),
                    "analysis_items": list(
                        record.get("analysis_report_items") or []
                    ),
                    "analysis_results": self._product_pdf_analysis_rows(
                        record.get("analysis_result_dict")
                    ),
                }
            )

        if not conditions:
            return None
        first_record = first_record or {}
        return {
            "group_id": str(group_id),
            "product_model": str(
                group.get("product_model")
                or first_record.get("product_model")
                or ""
            ),
            "barcode": str(
                group.get("barcode")
                or first_record.get("barcode")
                or ""
            ),
            "created_at": str(first_record.get("created_at") or ""),
            "test_time": str(
                group.get("time_text")
                or first_record.get("time_text")
                or ""
            ),
            "overall_result": str(overall_result or "-"),
            "conditions": conditions,
        }

    def _product_pdf_group_analysis_ready(self, group_id):
        group = self._product_pdf_group_data(group_id)
        if not isinstance(group, dict):
            return False

        records = group.get("records") or {}
        for index, condition in enumerate(self._product_condition_sequence()):
            key = self._product_condition_runtime_key(condition, index)
            record = records.get(key)
            if not isinstance(record, dict):
                return False
            state = str(record.get("analysis_report_state") or "")
            if state not in self._PRODUCT_PDF_TERMINAL_ANALYSIS_STATES:
                return False
        return True

    def _product_pdf_group_result_state(self, group_id):
        """Compatibility wrapper for the unified product result."""
        return self._product_group_result_state(group_id)

    def _try_export_product_test_pdf(self, group_id):
        complete, overall_result = self._product_group_result_state(group_id)
        if not complete:
            return None
        return self._maybe_export_product_test_pdf(group_id, overall_result)

    def _retry_product_pdf_export(self, group_id):
        group_key = str(group_id or "")
        retry_groups = getattr(
            self,
            "_product_pdf_manual_retry_groups",
            None,
        )
        if not isinstance(retry_groups, set):
            retry_groups = set()
            self._product_pdf_manual_retry_groups = retry_groups
        retry_groups.add(group_key)
        return self._try_export_product_test_pdf(group_id)

    def _maybe_export_product_test_pdf(self, group_id, overall_result):
        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        if not isinstance(report_config, dict) or not report_config.get("enabled", False):
            return None
        if not self._product_pdf_group_analysis_ready(group_id):
            return None

        report_data = self._build_product_pdf_report_data(group_id, overall_result)
        if not report_data:
            return None

        exported_states = getattr(self, "_product_pdf_report_states", None)
        if not isinstance(exported_states, dict):
            exported_states = {}
            self._product_pdf_report_states = exported_states
        paths = getattr(self, "_product_pdf_report_paths", None)
        if not isinstance(paths, dict):
            paths = {}
            self._product_pdf_report_paths = paths

        group_key = str(group_id)
        generation = int(
            getattr(self, "_product_pdf_report_generation", 0) or 0
        )
        job_key = (generation, group_key)
        jobs = getattr(self, "_product_pdf_export_jobs", None)
        if not isinstance(jobs, dict):
            jobs = {}
            self._product_pdf_export_jobs = jobs
        if job_key in jobs:
            pending = getattr(self, "_product_pdf_pending_exports", None)
            if not isinstance(pending, dict):
                pending = {}
                self._product_pdf_pending_exports = pending
            pending[job_key] = (
                copy.deepcopy(report_config),
                copy.deepcopy(report_data),
            )
            return None

        return self._start_product_pdf_export(
            job_key,
            report_config,
            report_data,
            previous_signature=exported_states.get(group_key),
            previous_path=paths.get(group_key),
        )

    def _start_product_pdf_export(
        self,
        job_key,
        report_config,
        report_data,
        *,
        previous_signature=None,
        previous_path=None,
    ):
        config_snapshot = copy.deepcopy(report_config)
        report_snapshot = copy.deepcopy(report_data)
        if not (
            isinstance(self, QObject)
            and QCoreApplication.instance() is not None
        ):
            signature, result = _execute_product_pdf_export(
                config_snapshot,
                report_snapshot,
                previous_signature,
                previous_path,
            )
            return self._handle_product_pdf_export_result(
                job_key,
                signature,
                result,
            )

        executor = getattr(self, "_product_pdf_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="product-pdf",
            )
            self._product_pdf_executor = executor
        jobs = getattr(self, "_product_pdf_export_jobs", None)
        if not isinstance(jobs, dict):
            jobs = {}
            self._product_pdf_export_jobs = jobs
        jobs[job_key] = executor.submit(
            _execute_product_pdf_export,
            config_snapshot,
            report_snapshot,
            previous_signature,
            previous_path,
        )
        self._ensure_product_pdf_poll_timer()
        return None

    def _ensure_product_pdf_poll_timer(self):
        timer = getattr(self, "_product_pdf_poll_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(100)
            timer.timeout.connect(self._poll_product_pdf_exports)
            self._product_pdf_poll_timer = timer
        if not timer.isActive():
            timer.start()

    def _poll_product_pdf_exports(self):
        jobs = getattr(self, "_product_pdf_export_jobs", None)
        if not isinstance(jobs, dict):
            jobs = {}
            self._product_pdf_export_jobs = jobs
        pending = getattr(self, "_product_pdf_pending_exports", None)
        if not isinstance(pending, dict):
            pending = {}
            self._product_pdf_pending_exports = pending

        for job_key, future in list(jobs.items()):
            if not future.done():
                continue
            jobs.pop(job_key, None)
            try:
                signature, result = future.result()
            except Exception as exc:
                signature = None
                result = ProductPdfExportResult(
                    ok=False,
                    message=f"PDF报告后台任务异常：{exc}",
                )
            self._handle_product_pdf_export_result(
                job_key,
                signature,
                result,
            )

            pending_request = pending.pop(job_key, None)
            current_generation = int(
                getattr(self, "_product_pdf_report_generation", 0) or 0
            )
            if pending_request and job_key[0] == current_generation:
                report_config, report_data = pending_request
                group_key = job_key[1]
                self._start_product_pdf_export(
                    job_key,
                    report_config,
                    report_data,
                    previous_signature=(
                        getattr(self, "_product_pdf_report_states", {}) or {}
                    ).get(group_key),
                    previous_path=(
                        getattr(self, "_product_pdf_report_paths", {}) or {}
                    ).get(group_key),
                )

        timer = getattr(self, "_product_pdf_poll_timer", None)
        if timer is not None and not jobs:
            timer.stop()

    def _handle_product_pdf_export_result(
        self,
        job_key,
        signature,
        result,
    ):
        generation, group_key = job_key
        current_generation = int(
            getattr(self, "_product_pdf_report_generation", 0) or 0
        )
        if generation != current_generation:
            return None

        if result.ok:
            retry_groups = getattr(
                self,
                "_product_pdf_manual_retry_groups",
                None,
            )
            is_manual_retry = (
                isinstance(retry_groups, set)
                and group_key in retry_groups
            )
            if is_manual_retry:
                retry_groups.discard(group_key)
            exported_states = getattr(self, "_product_pdf_report_states", None)
            if not isinstance(exported_states, dict):
                exported_states = {}
                self._product_pdf_report_states = exported_states
            if signature is not None:
                exported_states[group_key] = signature
            paths = getattr(self, "_product_pdf_report_paths", None)
            if not isinstance(paths, dict):
                paths = {}
                self._product_pdf_report_paths = paths
            if result.file_path:
                paths[group_key] = result.file_path
            logger = getattr(self, "default_logger", None)
            if logger is not None:
                logger.info(result.message)
            failures = getattr(self, "_product_pdf_report_failures", None)
            if isinstance(failures, dict):
                failures.pop(group_key, None)
            failure_dialog = getattr(
                self,
                "_product_pdf_failure_dialog",
                None,
            )
            if (
                failure_dialog is not None
                and getattr(
                    self,
                    "_product_pdf_failure_dialog_group_id",
                    None,
                )
                == group_key
            ):
                failure_dialog.close()
            if is_manual_retry:
                self._show_product_pdf_retry_success(
                    group_key,
                    result.file_path or result.message,
                )
            return result.file_path

        retry_groups = getattr(
            self,
            "_product_pdf_manual_retry_groups",
            None,
        )
        if isinstance(retry_groups, set):
            retry_groups.discard(group_key)
        logger = getattr(self, "default_logger", None)
        if logger is not None:
            logger.error(result.message)
        failures = getattr(self, "_product_pdf_report_failures", None)
        if not isinstance(failures, dict):
            failures = {}
            self._product_pdf_report_failures = failures
        failures[group_key] = result.message
        left_panel = getattr(self, "left_panel", None)
        set_stage = getattr(left_panel, "set_current_stage", None)
        if callable(set_stage):
            set_stage("测试完成，PDF报告生成失败", tone="ng")
        self._show_product_pdf_failure_retry(
            group_key,
            result.message,
        )
        return None

    def _show_product_pdf_failure_retry(self, group_id, message):
        if not (
            isinstance(self, QWidget)
            and QCoreApplication.instance() is not None
        ):
            return

        previous_dialog = getattr(
            self,
            "_product_pdf_failure_dialog",
            None,
        )
        if previous_dialog is not None:
            previous_dialog.close()

        dialog = QMessageBox(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("PDF报告生成失败")
        dialog.setText("产品测试已完成，但 PDF 报告生成失败。")
        dialog.setInformativeText(str(message or "未知错误"))
        retry_button = dialog.addButton(
            "重试生成",
            QMessageBox.AcceptRole,
        )
        dialog.addButton("关闭", QMessageBox.RejectRole)

        group_key = str(group_id or "")

        def handle_button_clicked(button):
            dialog.close()
            if button is retry_button:
                self._retry_product_pdf_export(group_key)

        def clear_dialog_reference(*_args):
            if getattr(self, "_product_pdf_failure_dialog", None) is dialog:
                self._product_pdf_failure_dialog = None
                self._product_pdf_failure_dialog_group_id = None

        dialog.buttonClicked.connect(handle_button_clicked)
        dialog.finished.connect(clear_dialog_reference)
        self._product_pdf_failure_dialog = dialog
        self._product_pdf_failure_dialog_group_id = group_key
        dialog.show()

    def _show_product_pdf_retry_success(self, group_id, file_path):
        if not (
            isinstance(self, QWidget)
            and QCoreApplication.instance() is not None
        ):
            return

        previous_dialog = getattr(
            self,
            "_product_pdf_success_dialog",
            None,
        )
        if previous_dialog is not None:
            previous_dialog.close()

        dialog = QMessageBox(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("PDF报告生成成功")
        dialog.setText("PDF 报告已重新生成。")
        dialog.setInformativeText(str(file_path or ""))
        dialog.addButton("关闭", QMessageBox.AcceptRole)

        group_key = str(group_id or "")

        def clear_dialog_reference(*_args):
            if getattr(self, "_product_pdf_success_dialog", None) is dialog:
                self._product_pdf_success_dialog = None
                self._product_pdf_success_dialog_group_id = None

        dialog.finished.connect(clear_dialog_reference)
        self._product_pdf_success_dialog = dialog
        self._product_pdf_success_dialog_group_id = group_key
        dialog.show()

    def _cancel_queued_product_pdf_exports(self):
        jobs = getattr(self, "_product_pdf_export_jobs", None)
        if not isinstance(jobs, dict):
            return 0
        pending = getattr(self, "_product_pdf_pending_exports", None)
        cancelled_count = 0
        for job_key, future in list(jobs.items()):
            if not future.cancel():
                continue
            jobs.pop(job_key, None)
            if isinstance(pending, dict):
                pending.pop(job_key, None)
            cancelled_count += 1

        timer = getattr(self, "_product_pdf_poll_timer", None)
        if timer is not None and not jobs:
            timer.stop()
        return cancelled_count

    def _reset_product_pdf_report_tracking(self):
        self._product_pdf_report_generation = (
            int(getattr(self, "_product_pdf_report_generation", 0) or 0) + 1
        )
        cancelled_count = self._cancel_queued_product_pdf_exports()
        logger = getattr(self, "default_logger", None)
        if cancelled_count and logger is not None:
            logger.info(
                f"product_pdf_cancelled_queued_jobs: {cancelled_count}"
            )
        failure_dialog = getattr(
            self,
            "_product_pdf_failure_dialog",
            None,
        )
        if failure_dialog is not None:
            failure_dialog.close()
        self._product_pdf_failure_dialog = None
        self._product_pdf_failure_dialog_group_id = None
        success_dialog = getattr(
            self,
            "_product_pdf_success_dialog",
            None,
        )
        if success_dialog is not None:
            success_dialog.close()
        self._product_pdf_success_dialog = None
        self._product_pdf_success_dialog_group_id = None
        self._product_pdf_manual_retry_groups = set()
        self._product_pdf_report_states = {}
        self._product_pdf_report_paths = {}
        self._product_pdf_report_failures = {}
        self._product_pdf_pending_exports = {}

    def _shutdown_product_pdf_exporter(self):
        jobs = getattr(self, "_product_pdf_export_jobs", None)
        while isinstance(jobs, dict) and jobs:
            for future in list(jobs.values()):
                try:
                    future.result()
                except Exception:
                    pass
            self._poll_product_pdf_exports()
            jobs = getattr(self, "_product_pdf_export_jobs", None)
        executor = getattr(self, "_product_pdf_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._product_pdf_executor = None
