"""Main-process ownership for automatic and manual spawned analysis tasks."""

from __future__ import annotations

from collections import deque
import json
import os
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox

from base.analysis_result_summary import failed_analysis_task_result
from base.analysis_service import AnalysisProcessService
from base.log_manager import LogManager
from consts import error_code
from ui.analysis_multichannel_result_window import (
    AnalysisMultichannelResultWindow,
)
from ui.sequence.analysis_task_builder import (
    AnalysisTaskBuildError,
    build_analysis_task_request,
)
from ui.sequence.analysis_report_snapshot import (
    build_analysis_report_items_from_task_result,
)


_MAIN_ANALYSIS_SUMMARY_EVENTS = frozenset(
    {
        "analysis_item_finished",
        "analysis_task_finished",
    }
)


def _format_analysis_main_log(record):
    event = str(record.get("event") or "analysis_event")
    common = [
        event,
        f"task_id={record.get('task_id', '')}",
        f"source={record.get('source', '')}",
        f"condition={record.get('condition_key', '')}",
    ]
    if event == "analysis_item_finished":
        fields = [
            f"item={record.get('config_key', '')}",
            f"type={record.get('analysis_type', '')}",
            f"channels={record.get('channel_count', 0)}",
            f"success={record.get('successful_channels', 0)}",
            f"failed={record.get('failed_channels', 0)}",
            f"duration_seconds={record.get('duration_seconds', 0)}",
        ]
        return " | ".join(common + fields)
    if event == "analysis_task_finished":
        fields = [
            f"execution={record.get('execution_status', '')}",
            f"result={record.get('final_judgement', '')}",
            f"duration_seconds={record.get('duration_seconds', 0)}",
        ]
        return " | ".join(common + fields)

    optional_fields = (
        ("config_key", "item"),
        ("analysis_type", "type"),
        ("runtime_key", "runtime"),
        ("config_item_name", "config_item"),
        ("error_type", "error_type"),
        ("error_message", "error"),
        ("artifact_path", "path"),
        ("duration_seconds", "duration_seconds"),
    )
    fields = [
        f"{label}={record[key]}"
        for key, label in optional_fields
        if record.get(key) not in (None, "")
    ]
    if event == "analysis_task_failed" and record.get("wav_path"):
        fields.append(f"wav={record['wav_path']}")
    return " | ".join(common + fields)


class SequenceWidgetAnalysisProcessOpsMixin:
    """Keep spawned analysis lifecycle out of recording and result widgets."""

    def _initialize_analysis_process_runtime(self):
        self._analysis_process_service = AnalysisProcessService()
        self._analysis_debug_logger = LogManager.set_log_handler("debug")
        self._analysis_task_queue = deque()
        self._analysis_task_records = {}
        self._analysis_manual_requested_at = {}
        self._analysis_manual_source_labels = {}
        self._analysis_pending_manual_view = None
        self._analysis_active_request = None
        self._analysis_handled_terminal_task_ids = set()
        self._analysis_round_completion_pending = False
        self._analysis_round_config_locked = False
        self._analysis_process_timer = QTimer(self)
        self._analysis_process_timer.setInterval(100)
        self._analysis_process_timer.timeout.connect(
            self._poll_analysis_process_runtime
        )
        self._analysis_process_timer.start()

    def _start_selected_condition_manual_analysis(self):
        if self._analysis_has_pending_tasks():
            QMessageBox.information(
                self,
                "提示",
                "还有分析任务未完成，请等待完成后再操作。",
            )
            return False
        if self._show_pending_manual_analysis_view():
            return True

        requested_at = time.monotonic()
        condition_key = self._selected_analysis_condition_key()
        record = self._resolve_condition_record(condition_key)
        wav_path = self._analysis_record_wav_path(record)
        if not wav_path:
            QMessageBox.information(self, "提示", "当前档位暂无可分析的录音。")
            self._refresh_analysis_action_state()
            return False
        if self._manual_analysis_target_is_recording(condition_key, wav_path):
            QMessageBox.information(
                self,
                "提示",
                "当前档位仍在录音，请选择已经录制完成的其他档位。",
            )
            self._refresh_analysis_action_state()
            return False
        source_label = self._analysis_condition_display_name(condition_key, record)
        try:
            request = self._build_process_analysis_request(
                condition_key,
                wav_path,
                "手动查看",
                record,
            )
        except AnalysisTaskBuildError as error:
            QMessageBox.warning(self, "分析不可用", str(error))
            return False
        self._analysis_task_records[request.task_id] = dict(record or {})
        self._analysis_manual_requested_at[request.task_id] = requested_at
        self._analysis_manual_source_labels[request.task_id] = source_label
        self._analysis_active_request = request
        try:
            pid = self._analysis_process_service.start(request)
        except (OSError, RuntimeError, ValueError) as error:
            self._analysis_active_request = None
            self._analysis_task_records.pop(request.task_id, None)
            self._analysis_manual_requested_at.pop(request.task_id, None)
            self._analysis_manual_source_labels.pop(request.task_id, None)
            self._analysis_pending_manual_view = {
                "result": None,
                "request": request,
                "source_label": source_label,
                "error_message": str(error) or "分析进程启动失败",
                "requested_at": requested_at,
                "ready_at": time.monotonic(),
            }
            self._set_manual_analysis_button_state(
                "failed",
                source_label=source_label,
            )
            self.default_logger.error(
                "manual_analysis_process_start_failed "
                f"task_id={request.task_id} condition={condition_key} "
                f"error={error}"
            )
            self._refresh_analysis_action_state()
            return False
        self._set_manual_analysis_button_state(
            "analyzing",
            completed=0,
            total=len(request.instances),
            source_label=source_label,
        )
        self.default_logger.info(
            "manual_analysis_process_started "
            f"task_id={request.task_id} condition={condition_key} "
            f"pid={pid} wav={request.wav_path}"
        )
        self._refresh_analysis_action_state()
        return True

    def _enqueue_automatic_analysis_current_recording(self):
        if not self._should_run_silent_analysis_after_recording():
            condition_key = self._get_active_product_condition_key()
            record = self._resolve_condition_record(condition_key)
            self._set_condition_analysis_stage(condition_key, "待判定", "pending")
            self._record_analysis_admission_state(
                record,
                state="not_required",
                status="未启用自动分析",
            )
            return True
        condition_key = self._get_active_product_condition_key()
        if not condition_key:
            condition_key = self._resolve_active_recording_waveform_direction(
                fallback=""
            )
        record = self._resolve_condition_record(condition_key)
        wav_path = self._analysis_record_wav_path(record)
        if not wav_path:
            self.default_logger.error(
                "automatic_analysis_enqueue_failed "
                f"condition={condition_key} reason=missing_wav"
            )
            self._set_condition_analysis_stage(
                condition_key,
                "分析失败",
                "ng",
            )
            self._record_analysis_admission_state(
                record,
                state="failed",
                status="分析失败",
                error="所选档位的 WAV 文件不存在",
            )
            return True
        try:
            request = self._build_process_analysis_request(
                condition_key,
                wav_path,
                "自动分析",
                record,
            )
        except AnalysisTaskBuildError as error:
            self.default_logger.error(
                "automatic_analysis_enqueue_failed "
                f"condition={condition_key} wav={wav_path} error={error}"
            )
            self._set_condition_analysis_stage(
                condition_key,
                "分析失败",
                "ng",
            )
            self._record_analysis_admission_state(
                record,
                state="failed",
                status="分析失败",
                error=str(error),
            )
            return True
        self._analysis_task_records[request.task_id] = dict(record or {})
        self._analysis_task_queue.append(request)
        self._set_condition_analysis_stage(condition_key, "分析排队", "running")
        self.default_logger.info(
            "automatic_analysis_enqueued "
            f"task_id={request.task_id} condition={condition_key} "
            f"queue_size={len(self._analysis_task_queue)} wav={request.wav_path}"
        )
        self._start_next_queued_analysis()
        self._refresh_analysis_action_state()
        return True

    def _record_analysis_admission_state(
        self,
        record,
        *,
        state,
        status,
        error="",
    ):
        if not isinstance(record, dict):
            return
        session_id = str(record.get("session_id") or "").strip()
        if not session_id:
            return
        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        report_enabled = bool(
            isinstance(report_config, dict)
            and report_config.get("enabled", False)
        )
        if state == "failed" and report_enabled:
            report_state = "failed"
            report_items = [
                {
                    "name": "分析报告",
                    "type": "",
                    "state": "failed",
                    "status": status,
                    "deviation": "-",
                    "error": str(error or status),
                    "image_errors": [],
                    "images": [],
                }
            ]
        else:
            report_state = "not_required"
            report_items = []
        recorded_info = dict(record.get("recorded_signal_info", {}) or {})
        recorded_info["labels"] = "not_labeled"
        self._update_recent_session(
            session_id,
            result_label=self._format_recent_session_result_label("not_labeled"),
            recorded_path=record.get("recorded_path"),
            recorded_signal_info=recorded_info,
            analysis_result_dict={},
            analysis_report_state=report_state,
            analysis_report_items=report_items,
        )

    def _build_process_analysis_request(
        self,
        condition_key,
        wav_path,
        source,
        record,
    ):
        info = dict((record or {}).get("recorded_signal_info", {}) or {})
        storage_snapshot = (
            dict(info.get("analysis_storage") or {})
            if source == "自动分析"
            else {}
        )
        if source == "自动分析":
            workspace = getattr(self, "channel_workspace", None)
            storage_snapshot["channel_labels"] = dict(
                getattr(workspace, "channel_layout", {}) or {}
            )
        fallback_factors = dict(
            getattr(self, "_live_mic_channel_v2pa_factors", {}) or {}
        )
        request = build_analysis_task_request(
            condition_key=condition_key,
            wav_path=wav_path,
            source=source,
            sequence_config=getattr(self, "sequence_config", []) or [],
            analysis_config=getattr(self, "analysis_config", {}) or {},
            storage_snapshot=storage_snapshot,
            saved_active_input_channels=info.get("active_input_channels"),
            fallback_v2pa_factors=fallback_factors,
        )
        missing_channels = sorted(
            {
                instance.raw_channel
                for instance in request.instances
                if not instance.calibration_available
            }
        )
        if missing_channels:
            channel_text = "、".join(
                f"In{raw_channel + 1}" for raw_channel in missing_channels
            )
            self.default_logger.warning(
                "analysis_calibration_missing "
                f"task_id={request.task_id} condition={condition_key} "
                f"channels={channel_text} wav={request.wav_path}"
            )
            QMessageBox.warning(
                self,
                "输入通道未校准",
                f"{channel_text} 未找到有效校准系数，将按现有默认系数继续分析，"
                "结果仅供参考。",
            )
        return request

    def _start_next_queued_analysis(self):
        service = getattr(self, "_analysis_process_service", None)
        if service is None or service.active or self._analysis_active_request is not None:
            return False
        queue = getattr(self, "_analysis_task_queue", None)
        if not queue:
            self._finish_deferred_analysis_round_if_ready()
            self._refresh_analysis_action_state()
            return False
        request = queue.popleft()
        self._analysis_active_request = request
        self._set_condition_analysis_stage(
            request.condition_key,
            "分析中",
            "running",
        )
        try:
            pid = service.start(request)
        except (OSError, RuntimeError, ValueError) as error:
            from base.analysis_process_protocol import AnalysisWorkerFailure

            failure = AnalysisWorkerFailure(
                request.task_id,
                "启动子进程",
                type(error).__name__,
                str(error) or type(error).__name__,
            )
            self._handle_analysis_terminal(
                failed_analysis_task_result(request, failure)
            )
            self._analysis_handled_terminal_task_ids.discard(request.task_id)
            self._analysis_active_request = None
            QTimer.singleShot(0, self._start_next_queued_analysis)
            return False
        self.default_logger.info(
            "automatic_analysis_process_started "
            f"task_id={request.task_id} condition={request.condition_key} "
            f"pid={pid} remaining_queue={len(queue)} wav={request.wav_path}"
        )
        return True

    def _poll_analysis_process_runtime(self):
        service = getattr(self, "_analysis_process_service", None)
        if service is None:
            return
        events, log_records = service.poll()
        for record in log_records:
            self._write_analysis_worker_log(record)
        active_request = self._analysis_active_request
        for kind, payload in events:
            if kind == "progress" and active_request is not None:
                if active_request.source == "自动分析":
                    self._set_condition_analysis_stage(
                        active_request.condition_key,
                        payload.stage,
                        "running",
                    )
                elif active_request.source == "手动查看":
                    source_label = self._analysis_manual_source_labels.get(
                        active_request.task_id,
                        active_request.condition_key,
                    )
                    self._set_manual_analysis_button_state(
                        "analyzing",
                        completed=payload.completed_instances,
                        total=payload.total_instances,
                        source_label=source_label,
                    )
            elif kind == "result":
                self._handle_analysis_terminal(payload)
            elif kind == "failure" and active_request is not None:
                self._handle_analysis_terminal(
                    failed_analysis_task_result(active_request, payload)
                )
        if not service.active and self._analysis_active_request is not None:
            finished_request = self._analysis_active_request
            round_completion_pending = bool(
                getattr(self, "_analysis_round_completion_pending", False)
            )
            task_id = finished_request.task_id
            if task_id not in self._analysis_handled_terminal_task_ids:
                from base.analysis_process_protocol import AnalysisWorkerFailure

                failure = AnalysisWorkerFailure(
                    task_id,
                    "子进程退出",
                    "MissingResult",
                    "分析子进程没有返回最终结果",
                )
                self._handle_analysis_terminal(
                    failed_analysis_task_result(
                        self._analysis_active_request,
                        failure,
                    )
                )
            self._analysis_active_request = None
            self._analysis_handled_terminal_task_ids.discard(task_id)
            next_started = self._start_next_queued_analysis()
            if (
                finished_request.source == "自动分析"
                and not next_started
                and not round_completion_pending
            ):
                self._restore_waiting_stage_after_automatic_analysis()
        self._refresh_analysis_action_state()

    def _handle_analysis_terminal(self, result):
        if result.task_id in self._analysis_handled_terminal_task_ids:
            return
        self._analysis_handled_terminal_task_ids.add(result.task_id)
        record = self._analysis_task_records.pop(result.task_id, {})
        if result.source == "手动查看":
            requested_at = getattr(
                self,
                "_analysis_manual_requested_at",
                {},
            ).pop(result.task_id, None)
            source_labels = getattr(self, "_analysis_manual_source_labels", {})
            source_label = source_labels.pop(result.task_id, "") or (
                self._analysis_condition_display_name(
                    result.condition_key,
                    record,
                )
            )
            ready_at = time.monotonic()
            self._analysis_pending_manual_view = {
                "result": result,
                "request": getattr(self, "_analysis_active_request", None),
                "source_label": source_label,
                "error_message": result.error_message,
                "requested_at": requested_at,
                "ready_at": ready_at,
            }
            if result.execution_status == "分析完成" and result.instance_results:
                state = "completed"
            else:
                state = "failed"
            self._set_manual_analysis_button_state(
                state,
                source_label=source_label,
            )
            duration = (
                ready_at - requested_at
                if requested_at is not None
                else 0.0
            )
            self.default_logger.info(
                "manual_analysis_result_ready "
                f"task_id={result.task_id} condition={result.condition_key} "
                f"execution={result.execution_status} "
                f"duration_seconds={duration:.3f}"
            )
            return
        self._apply_automatic_analysis_result(result, record)

    def _apply_automatic_analysis_result(self, result, record):
        if (
            result.execution_status == "分析完成"
            and result.judgement_status == "已判定"
            and result.final_judgement in {"OK", "NG"}
        ):
            label = result.final_judgement
            display_text = label
            tone = "ok" if label == "OK" else "ng"
        elif result.execution_status == "结果不完整":
            label = "not_labeled"
            display_text = "结果不完整"
            tone = "ng"
        elif result.execution_status == "分析失败":
            label = "not_labeled"
            display_text = "分析失败"
            tone = "ng"
        else:
            label = "not_labeled"
            display_text = "未产生判定"
            tone = "pending"

        artifact_failures = [
            artifact
            for item in result.instance_results
            for artifact in item.artifacts
            if artifact.status == "保存失败"
        ]
        if artifact_failures:
            self.default_logger.error(
                "automatic_analysis_artifact_save_failed "
                f"task_id={result.task_id} condition={result.condition_key} "
                f"count={len(artifact_failures)}"
            )

        updated_record = self._update_analysis_record_label(
            result.condition_key,
            result.wav_path,
            record,
            label,
        )
        self._update_process_result_session_snapshot(
            result,
            updated_record or record,
            label,
        )
        self._sync_process_result_to_condition_panel(result, display_text, tone)
        if artifact_failures:
            self._show_analysis_artifact_warning(result.condition_key)
        self.default_logger.info(
            "automatic_analysis_applied "
            f"task_id={result.task_id} condition={result.condition_key} "
            f"execution={result.execution_status} "
            f"judgement_status={result.judgement_status} "
            f"result={result.final_judgement or ''} wav={result.wav_path}"
        )
        if result.execution_status != "分析完成":
            self.default_logger.error(
                "automatic_analysis_incomplete "
                f"task_id={result.task_id} condition={result.condition_key} "
                f"stage={result.error_stage or ''} "
                f"error_type={result.error_type or ''} "
                f"error={result.error_message or ''} wav={result.wav_path}"
            )

    def _show_analysis_artifact_warning(self, condition_key):
        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return
        set_details = getattr(left_panel, "set_condition_analysis_details", None)
        if callable(set_details):
            set_details(
                condition_key,
                {"结果文件": "保存失败，请查看日志"},
            )
        left_panel.set_current_stage("结果文件保存失败", tone="ng")

    def _update_analysis_record_label(
        self,
        condition_key,
        wav_path,
        record,
        label,
    ):
        recorded_info = dict((record or {}).get("recorded_signal_info", {}) or {})
        if not recorded_info:
            self.default_logger.error(
                "automatic_analysis_label_update_failed "
                f"condition={condition_key} wav={wav_path} reason=missing_record_metadata"
            )
            return None
        code, message, stable_path, updated_info = self._relabel_stored_audio_record(
            wav_path,
            recorded_info,
            label,
        )
        if code != error_code.OK:
            self.default_logger.error(
                "automatic_analysis_label_update_failed "
                f"condition={condition_key} wav={wav_path} label={label} "
                f"error={message}"
            )
            return None
        updated_record = dict(record or {})
        updated_record["recorded_path"] = stable_path
        updated_record["recorded_signal_info"] = dict(updated_info or {})
        cache = getattr(self, "_condition_record_cache", None)
        if isinstance(cache, dict):
            current = cache.get(condition_key)
            current_path = self._analysis_record_wav_path(current)
            if current_path and os.path.normcase(current_path) == os.path.normcase(
                os.path.abspath(wav_path)
            ):
                cache[condition_key] = updated_record
        current_path = self._resolve_audio_path_to_abs(
            getattr(self, "recorded_path", None)
        )
        if current_path and os.path.normcase(current_path) == os.path.normcase(
            os.path.abspath(wav_path)
        ):
            self.recorded_path = stable_path
            self.recorded_signal_info = dict(updated_info or {})
        return updated_record

    def _update_process_result_session_snapshot(self, result, record, label):
        if not isinstance(record, dict):
            return
        session_id = str(record.get("session_id") or "").strip()
        if not session_id:
            return
        active_request = self._analysis_active_request
        analysis_config = (
            active_request.analysis_config_snapshot.to_dict()
            if active_request is not None
            and active_request.task_id == result.task_id
            else getattr(self, "analysis_config", {}) or {}
        )
        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        if isinstance(report_config, dict) and report_config.get("enabled", False):
            try:
                if not result.instance_results and result.execution_status != "分析完成":
                    report_items = [
                        {
                            "name": "分析报告",
                            "type": "",
                            "state": "failed",
                            "status": result.execution_status,
                            "deviation": "-",
                            "error": result.error_message or "分析子进程未返回结果",
                            "image_errors": [],
                            "images": [],
                        }
                    ]
                else:
                    report_items = build_analysis_report_items_from_task_result(
                        result,
                        analysis_config,
                    )
                report_state = (
                    "failed"
                    if any(item.get("state") == "failed" for item in report_items)
                    else "completed"
                )
            except (OSError, TypeError, ValueError) as error:
                self.default_logger.error(
                    "analysis_report_snapshot_failed "
                    f"task_id={result.task_id} error={error}"
                )
                report_items = [
                    {
                        "name": "分析报告",
                        "type": "",
                        "state": "failed",
                        "status": "分析失败",
                        "deviation": "-",
                        "error": str(error),
                        "image_errors": [],
                        "images": [],
                    }
                ]
                report_state = "failed"
        else:
            report_items = []
            report_state = "not_required"
        recorded_info = dict(record.get("recorded_signal_info", {}) or {})
        recorded_info["labels"] = label
        analysis_result_dict = {
            item.runtime_key: (
                item.judgement == "OK",
                0.0,
            )
            for item in result.instance_results
            if item.execution_status == "分析完成"
            and item.contributes_to_final
            and item.judgement in {"OK", "NG"}
        }
        self._update_recent_session(
            session_id,
            result_label=self._format_recent_session_result_label(label),
            recorded_path=record.get("recorded_path") or result.wav_path,
            recorded_signal_info=recorded_info,
            analysis_result_dict=analysis_result_dict,
            analysis_report_state=report_state,
            analysis_report_items=report_items,
        )

    def _sync_process_result_to_condition_panel(self, result, display_text, tone):
        channel_results = self._build_process_channel_results(result)
        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            set_channels = getattr(left_panel, "set_channels", None)
            if callable(set_channels):
                set_channels(sorted({item["raw_channel"] for item in channel_results}))
            set_channel_results = getattr(
                left_panel,
                "set_condition_channel_results",
                None,
            )
            if callable(set_channel_results):
                set_channel_results(result.condition_key, channel_results)
            left_panel.set_condition_result(
                result.condition_key,
                display_text,
                tone=tone,
            )
        workspace = getattr(self, "channel_workspace", None)
        set_context = getattr(workspace, "set_condition_context", None)
        if callable(set_context):
            workspace_status = (
                "分析完成"
                if result.execution_status == "分析完成"
                else display_text
            )
            set_context(result.condition_key, status=workspace_status)
        if (
            workspace is not None
            and result.final_judgement in {"OK", "NG"}
            and hasattr(workspace, "set_condition_result")
        ):
            workspace.set_condition_result(
                result.condition_key,
                result.final_judgement,
            )

    @staticmethod
    def _build_process_channel_results(task_result):
        column_names = {
            "SPL": "SPL",
            "FBA": "FBA",
            "FFT": "FFT",
            "AI": "AI分析",
            "Spec": "Spec",
        }
        channels = {}
        for item in task_result.instance_results:
            channel = channels.setdefault(
                int(item.raw_channel),
                {"columns": {}, "judgements": []},
            )
            if item.execution_status != "分析完成":
                value = "分析失败"
            elif item.contributes_to_final:
                value = item.judgement or "未判定"
                channel["judgements"].append(item.judgement or "")
            else:
                value = "分析完成"
            column = column_names.get(item.analysis_type, item.analysis_type)
            channel["columns"].setdefault(column, []).append(value)
        output = []
        for raw_channel, state in sorted(channels.items()):
            verdicts = state["judgements"]
            if "NG" in verdicts:
                overall = "NG"
            elif verdicts and all(value == "OK" for value in verdicts):
                overall = "OK"
            else:
                overall = "待判定"
            row = {"raw_channel": raw_channel, "result": overall, "details": {}}
            for column, values in state["columns"].items():
                if "分析失败" in values:
                    value = "分析失败"
                elif "NG" in values:
                    value = "NG"
                elif values and all(item == "OK" for item in values):
                    value = "OK"
                elif values and all(item == "分析完成" for item in values):
                    value = "分析完成"
                else:
                    value = "未判定"
                row[column] = value
                row["details"][column] = "；".join(values)
            output.append(row)
        return output

    def _show_pending_manual_analysis_view(self):
        pending = getattr(self, "_analysis_pending_manual_view", None)
        if not pending:
            return False

        result = pending.get("result")
        source_label = str(pending.get("source_label") or "").strip()
        requested_at = pending.get("requested_at")
        ready_at = pending.get("ready_at")
        window_count = 0
        if result is not None and result.instance_results:
            window_started_at = time.monotonic()
            window_count = self._show_manual_analysis_result_windows(
                result,
                request=pending.get("request"),
                source_label=source_label,
            )
            viewed_at = time.monotonic()
            log_message = (
                "manual_analysis_windows_opened "
                f"task_id={result.task_id} condition={result.condition_key} "
                f"window_count={window_count}"
            )
            if requested_at is not None and ready_at is not None:
                log_message += (
                    f" total_wait_seconds={viewed_at - requested_at:.3f}"
                    f" ready_wait_seconds={viewed_at - ready_at:.3f}"
                    f" window_build_seconds={viewed_at - window_started_at:.3f}"
                )
            self.default_logger.info(log_message)
        else:
            error_message = str(
                pending.get("error_message")
                or getattr(result, "error_message", "")
                or "分析子进程未返回可显示的结果。"
            )
            QMessageBox.warning(self, "分析失败", error_message)

        self._analysis_pending_manual_view = None
        self._set_manual_analysis_button_state("idle")
        self._refresh_analysis_action_state()
        return True

    def _show_manual_analysis_result_windows(
        self,
        result,
        *,
        request=None,
        source_label="",
    ):
        close_windows = getattr(self, "_close_analysis_windows", None)
        if callable(close_windows):
            close_windows()
        self.analysis_window = []
        grouped = {}
        for item in result.instance_results:
            grouped.setdefault(item.config_key, []).append(item)
        request = request or self._analysis_active_request
        config_snapshot = (
            request.analysis_config_snapshot.to_dict()
            if request is not None
            else getattr(self, "analysis_config", {}) or {}
        )
        channel_labels = dict(
            getattr(
                getattr(self, "channel_workspace", None),
                "channel_layout",
                {},
            )
            or {}
        )
        for config_key, instance_results in grouped.items():
            window = AnalysisMultichannelResultWindow(
                config_key,
                instance_results,
                config=config_snapshot.get(config_key, {}),
                channel_labels=channel_labels,
                source_label=source_label,
            )
            window.setAttribute(Qt.WA_DeleteOnClose, True)
            window.destroyed.connect(
                lambda _object=None, target=window: (
                    self._discard_manual_analysis_window(target)
                )
            )
            window.show()
            window.raise_()
            self.analysis_window.append(window)
        return len(grouped)

    def _discard_manual_analysis_window(self, window):
        self.analysis_window = [
            item
            for item in (getattr(self, "analysis_window", []) or [])
            if item is not window
        ]

    def _selected_analysis_condition_key(self):
        left_panel = getattr(self, "left_panel", None)
        selected = str(
            getattr(left_panel, "selected_condition_key", "") or ""
        ).strip()
        if selected:
            return selected
        active = self._get_active_product_condition_key()
        if active:
            return active
        resolve_direction = getattr(self, "_resolve_waveform_direction", None)
        if callable(resolve_direction):
            return str(resolve_direction(fallback="") or "").strip()
        return ""

    def _analysis_record_wav_path(self, record):
        if not isinstance(record, dict):
            return None
        info = dict(record.get("recorded_signal_info", {}) or {})
        for value in (record.get("recorded_path"), info.get("file_path")):
            path = self._resolve_audio_path_to_abs(value)
            if path and os.path.isfile(path):
                return path
        return None

    def _analysis_condition_display_name(self, condition_key, record=None):
        key = str(condition_key or "").strip()
        record = dict(record or {}) if isinstance(record, dict) else {}
        snapshot = record.get("config_snapshot")
        condition = {}
        if isinstance(snapshot, dict):
            candidate = snapshot.get("condition_config")
            if isinstance(candidate, dict):
                condition = candidate
        if not condition:
            resolve_condition = getattr(
                self,
                "_resolve_recent_session_condition",
                None,
            )
            if callable(resolve_condition):
                candidate = resolve_condition(key)
                if isinstance(candidate, dict):
                    condition = candidate

        port_name = str(condition.get("group_name") or "").strip()
        condition_name = str(
            condition.get("condition_name")
            or condition.get("name")
            or key
        ).strip()
        if port_name and condition_name:
            return f"{port_name} / {condition_name}"

        mode_text = str(record.get("mode_text") or "").strip()
        return mode_text or condition_name or key or "所选档位"

    def _manual_analysis_target_is_recording(self, condition_key, wav_path):
        if not (
            getattr(self, "player_status_flag", False)
            or getattr(self, "_record_workflow_busy", False)
        ):
            return False

        key = str(condition_key or "").strip()
        get_active_key = getattr(self, "_get_active_product_condition_key", None)
        active_key = str(get_active_key() or "").strip() if callable(get_active_key) else ""
        if not active_key:
            resolve_active = getattr(
                self,
                "_resolve_active_recording_waveform_direction",
                None,
            )
            if callable(resolve_active):
                active_key = str(resolve_active(fallback="") or "").strip()
        if active_key and key == active_key:
            return True

        current_wav_path = str(getattr(self, "recorded_path", "") or "").strip()
        if not current_wav_path or not wav_path:
            return False
        return os.path.normcase(os.path.abspath(current_wav_path)) == os.path.normcase(
            os.path.abspath(wav_path)
        )

    def _set_manual_analysis_button_state(
        self,
        state,
        *,
        completed=0,
        total=1,
        source_label="",
    ):
        button = getattr(self, "data_btn", None)
        if button is None:
            return
        if state == "analyzing":
            setter = getattr(button, "set_analyzing", None)
            if callable(setter):
                setter(completed, total, source_label)
        elif state == "completed":
            setter = getattr(button, "set_completed", None)
            if callable(setter):
                setter(source_label)
        elif state == "failed":
            setter = getattr(button, "set_failed", None)
            if callable(setter):
                setter(source_label)
        else:
            setter = getattr(button, "set_idle", None)
            if callable(setter):
                setter()

    def _analysis_has_pending_tasks(self):
        service = getattr(self, "_analysis_process_service", None)
        queue = getattr(self, "_analysis_task_queue", ())
        return bool(
            (service is not None and service.active)
            or self._analysis_active_request is not None
            or queue
        )

    def _restore_waiting_stage_after_automatic_analysis(self):
        if self._analysis_has_pending_tasks():
            return False
        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return False
        if (
            getattr(self, "player_status_flag", False)
            or getattr(self, "_record_workflow_busy", False)
        ):
            left_panel.clear_current_stage()
            return True
        left_panel.set_current_stage("等待下一档位", tone="pending")
        return True

    def _set_condition_analysis_stage(
        self,
        condition_key,
        text,
        tone,
        *,
        detail_text=None,
    ):
        key = str(condition_key or "").strip()
        if not key:
            return
        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            left_panel.set_condition_result(key, text, tone=tone)
            left_panel.set_current_stage(detail_text or text, tone=tone)
        workspace = getattr(self, "channel_workspace", None)
        set_context = getattr(workspace, "set_condition_context", None)
        if callable(set_context):
            set_context(key, status=text)

    def _refresh_analysis_action_state(self):
        button = getattr(self, "data_btn", None)
        if button is None:
            return
        condition_key = self._selected_analysis_condition_key()
        record = self._resolve_condition_record(condition_key) if condition_key else None
        wav_path = self._analysis_record_wav_path(record)
        has_pending_tasks = self._analysis_has_pending_tasks()
        if getattr(self, "_analysis_pending_manual_view", None):
            enabled = not has_pending_tasks
        else:
            enabled = (
                bool(wav_path)
                and not has_pending_tasks
                and not self._manual_analysis_target_is_recording(
                    condition_key,
                    wav_path,
                )
            )
        button.setEnabled(enabled)

    def _defer_round_completion_for_analysis(self):
        if not self._analysis_has_pending_tasks():
            return False
        self._analysis_round_completion_pending = True
        self._lock_analysis_round_config()
        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            left_panel.set_current_stage("本轮录音完成，等待分析", tone="running")
        return True

    def _finish_deferred_analysis_round_if_ready(self):
        if (
            not getattr(self, "_analysis_round_completion_pending", False)
            or self._analysis_has_pending_tasks()
        ):
            return False
        self._analysis_round_completion_pending = False
        deferred_serial_close = bool(
            getattr(self, "_analysis_deferred_serial_close", False)
        )
        self._analysis_deferred_serial_close = False
        round_text, round_tone, round_resolved = (
            self._automatic_round_display_state()
        )
        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            if round_resolved:
                left_panel.set_final_result(round_text, tone=round_tone)
                stage_text = (
                    "本轮完成"
                    if round_text in ("OK", "NG")
                    else "本轮分析完成，未判定"
                )
                left_panel.set_current_stage(stage_text, tone=round_tone)
            else:
                left_panel.set_current_stage(
                    "本轮分析完成，未判定",
                    tone="pending",
                )
        if deferred_serial_close:
            self._serial_product_waiting_for_close = True
        else:
            self._manual_product_condition_group_id = ""
            self._current_cycle_recorded_count = None
            unlock_barcode = getattr(self, "_unlock_sn_for_product_round", None)
            if callable(unlock_barcode):
                unlock_barcode(clear=True)
            self._unlock_analysis_round_config()
        self.update_player_btn_is_paused()
        return True

    def _lock_analysis_round_config(self):
        self._analysis_round_config_locked = True
        combo = getattr(self, "using_file_combobox", None)
        if combo is not None:
            combo.setEnabled(False)

    def _unlock_analysis_round_config(self):
        self._analysis_round_config_locked = False
        combo = getattr(self, "using_file_combobox", None)
        if combo is not None:
            combo.setEnabled(True)

    def _write_analysis_worker_log(self, record):
        if not isinstance(record, dict):
            return
        level = str(record.get("level") or "INFO").upper()
        raw_message = json.dumps(record, ensure_ascii=False, default=str)
        debug_writer = getattr(
            getattr(self, "_analysis_debug_logger", None),
            "info",
            None,
        )
        if callable(debug_writer):
            debug_writer(raw_message)

        event = str(record.get("event") or "")
        if (
            event not in _MAIN_ANALYSIS_SUMMARY_EVENTS
            and level not in {"ERROR", "WARNING"}
        ):
            return
        writer = getattr(
            self.default_logger,
            {
                "ERROR": "error",
                "WARNING": "warning",
                "DEBUG": "debug",
            }.get(level, "info"),
            None,
        )
        if callable(writer):
            writer(_format_analysis_main_log(record))
