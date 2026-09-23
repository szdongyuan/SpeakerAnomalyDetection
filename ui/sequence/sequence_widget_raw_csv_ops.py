"""CSV service attachment and presentation, separate from recording lifecycle."""
import copy
import os
from dataclasses import dataclass, field
from uuid import uuid4
from collections import deque
from time import perf_counter

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY

from base.raw_audio_csv_protocol import CsvFailure, CsvExportRequest
from base.raw_audio_csv_zip import raw_csv_zip_path
from base.analysis_artifact_paths import build_raw_audio_csv_path, storage_context_from_metadata


def record_gui_stage(host, stage, started, *, request=None):
    """Retain only recent small parent-clock observations; never log on Qt."""
    finished = perf_counter()
    timings = getattr(host, "_recording_stage_timings", None)
    if timings is None:
        timings = host._recording_stage_timings = deque(maxlen=64)
    request = request or getattr(host, "_recording_process_request", None)
    timings.append(dict(stage=stage, request_id=getattr(request, "request_id", ""),
                        started=started, finished=finished, seconds=finished - started))


@dataclass
class CsvRecordingAdmission:
    recording_id: str
    csv_enabled_snapshot: bool
    csv_reservation: object = None
    scope_owner: object = field(default=None, repr=False)
    cancelled: bool = False


class CsvRecordingAdmissionScope:
    """Roll back only this attempt until a recording context takes ownership."""
    def __init__(self, host):
        self.host = host
        self.admission = None
        self.keep = False
        self.allowed = True

    def __enter__(self):
        reserve = getattr(self.host, "_reserve_raw_audio_csv_recording", None)
        if callable(reserve):
            # Qt can deliver another start while the admitted invocation is
            # initializing capture, before its context exists. That invocation
            # must not acquire cleanup responsibility for the pending token.
            if (any(bool(getattr(self.host, name, False)) for name in (
                    "_record_workflow_busy", "player_status_flag",
                    "_recording_publication_in_progress"))
                    or getattr(self.host, "_recording_process_contexts", None)):
                self.allowed = False
                return self
            self.admission = reserve()
            self.allowed = self.admission is not None
            if self.admission is not None and self.admission.scope_owner is None:
                self.admission.scope_owner = self
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.admission is None or self.admission.scope_owner is not self:
            return
        # A successful preparation leaves the token pending, but its scope has
        # ended. The next independent startup claims rollback responsibility.
        # Nested prepare/start/judge scopes only borrow while this owner lives.
        self.admission.scope_owner = None
        if exc_type is not None or not self.keep:
            self.host._release_raw_audio_csv_recording(self.admission)


class SequenceWidgetRawCsvOpsMixin:
    def _initialize_raw_audio_csv_runtime(self, *, raw_audio_csv_bridge=None):
        self._owned_raw_audio_csv_tasks = set()
        self._presented_raw_audio_csv_tasks = set()
        self._raw_audio_csv_recording_paths = set()
        self._pending_raw_audio_csv_recording = None
        self._owns_raw_audio_csv_service = raw_audio_csv_bridge is None
        if raw_audio_csv_bridge is None:
            from base.raw_audio_csv_service import RawAudioCsvService
            from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
            raw_audio_csv_bridge = RawAudioCsvServiceBridge(RawAudioCsvService(), self)
            QApplication.instance().aboutToQuit.connect(raw_audio_csv_bridge.begin_shutdown)
        self.raw_audio_csv_bridge = raw_audio_csv_bridge
        self.raw_audio_csv_service = raw_audio_csv_bridge.service
        self._raw_audio_csv_subscription = raw_audio_csv_bridge.subscribe(
            self._on_raw_audio_csv_service_event, owner=self)

    def _raw_audio_csv_snapshot(self):
        return self.raw_audio_csv_service.snapshot()

    def _on_raw_audio_csv_service_event(self, event):
        self._refresh_raw_audio_csv_admission()
        presented = getattr(self, "_presented_raw_audio_csv_tasks", None)
        if presented is None:
            presented = self._presented_raw_audio_csv_tasks = set()
        if event.kind == "released":
            self._owned_raw_audio_csv_tasks.discard(event.task.request.task_id)
            presented.discard(event.task.request.task_id)
        timer = getattr(self, "_raw_audio_csv_close_timer", None)
        if timer is not None and timer.isActive() and event.kind in ("released", "closed"):
            self._resume_raw_audio_csv_close()
        if event.kind != "terminal":
            return
        task_id = event.task.request.task_id
        if task_id not in self._owned_raw_audio_csv_tasks or task_id in presented:
            return
        presented.add(task_id)
        # Preserve original recording attribution; never consult the current run.
        if isinstance(event.result, CsvFailure):
            if event.result.stage in {"zip_write", "zip_verify", "zip_publish", "csv_cleanup"}:
                self.default_logger.error(
                    f"raw_audio_csv_zip_failed path={event.task.request.csv_path} "
                    f"stage={event.result.stage} error={event.result.message}")
                QMessageBox.warning(self, "原始音频 ZIP 保存失败",
                    f"CSV 已保存，但 ZIP 压缩失败。\n{event.task.request.csv_path}\n"
                    f"{event.result.message}")
            else:
                self._on_raw_audio_csv_export_failed(
                    event.task.request.wav_path, event.result.message)
        else:
            self._on_raw_audio_csv_export_succeeded(event.result.archive_path)
            if event.result.csv_retained or event.result.cleanup_diagnostics:
                details = "\n".join(event.result.cleanup_diagnostics)
                self.default_logger.warning(
                    f"raw_audio_csv_cleanup_failed path={event.result.csv_path} error={details}")
                QMessageBox.warning(self, "原始音频 CSV 清理失败",
                    f"ZIP 已保存但 CSV 清理失败。\nZIP：{event.result.archive_path}\n"
                    f"CSV：{event.result.csv_path}\n{details}")

    def _raw_audio_csv_admission_reason(self):
        pending = getattr(self, "_pending_raw_audio_csv_recording", None)
        if pending is not None:
            return ""
        enabled = (getattr(self, "product_test_project_context", {}) or {}).get(
            EXPORT_RAW_AUDIO_CSV_KEY, False) is True
        if not enabled:
            return ""
        snapshot = self.raw_audio_csv_service.snapshot()
        if snapshot.phase != "open":
            return "原始音频 CSV 服务正在关闭或不可用，请稍后重试。"
        if snapshot.outstanding >= snapshot.capacity:
            return "原始音频 CSV 保存队列已满（16 份），请等待保存完成后重新触发录音。"
        return ""

    def _csv_admission_notice(self, reason):
        self.default_logger.info(f"recording_rejected_csv_busy reason={reason}")
        if not (getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", False):
            QMessageBox.warning(self, "录音暂不可用", reason)

    def _reserve_raw_audio_csv_recording(self):
        if getattr(self, "_close_in_progress", False) or getattr(self, "_recording_closed", False):
            return None
        pending = getattr(self, "_pending_raw_audio_csv_recording", None)
        if pending is not None:
            return pending
        recording_id = uuid4().hex
        enabled = (getattr(self, "product_test_project_context", {}) or {}).get(
            EXPORT_RAW_AUDIO_CSV_KEY, False) is True
        token = None
        if enabled:
            admission = self.raw_audio_csv_service.reserve(recording_id)
            if admission.status != "accepted":
                reason = ("原始音频 CSV 保存队列已满（16 份），请等待保存完成后重新触发录音。"
                          if admission.status == "full" else
                          "原始音频 CSV 服务正在关闭或不可用，请稍后重试。")
                self._csv_admission_notice(reason)
                return None
            token = admission.reservation
        pending = CsvRecordingAdmission(recording_id, enabled, token)
        self._pending_raw_audio_csv_recording = pending
        return pending

    def _release_raw_audio_csv_recording(self, admission):
        if getattr(self, "_pending_raw_audio_csv_recording", None) is not admission:
            return
        self._pending_raw_audio_csv_recording = None
        admission.cancelled = True
        if admission.csv_reservation is not None:
            self.raw_audio_csv_service.release_reservation(admission.csv_reservation)
            admission.csv_reservation = None

    def _release_raw_audio_csv_context(self, context):
        token = context.csv_reservation
        context.csv_reservation = None
        if token is not None:
            self.raw_audio_csv_service.release_reservation(token)

        permit = context.csv_path_permit
        context.csv_path_permit = None
        if permit is not None:
            bridge = getattr(self, "recording_bridge", None)
            service = bridge.service if bridge is not None else None
            path = context.request.path
            if service is not None and service.is_path_leased(path):
                # Failed cleanup / optional release warning can outlive the UI
                # context and even a completed CSV. Keep its exact claim until
                # the recording lease is gone; never rely on a late Qt callback.
                self.raw_audio_csv_service.release_mutation(
                    permit, ready=lambda: not service.is_path_leased(path))
            else:
                self.raw_audio_csv_service.release_mutation(permit)
                # CSV task ownership separately lasts through RELEASED. Only
                # unresolved recording leases need the extra local path gate.
                getattr(self, "_raw_audio_csv_recording_paths", set()).discard(path)

    def _refresh_raw_audio_csv_admission(self):
        if getattr(self, "_close_in_progress", False):
            return
        if getattr(self, "player_status_flag", False) or getattr(self, "_record_workflow_busy", False):
            return
        refresh = getattr(self, "update_player_btn_is_paused", None)
        if callable(refresh):
            refresh()

    def _begin_raw_audio_csv_close(self, *, application_exit=False):
        """Close local recording admission before any asynchronous exit boundary."""
        if application_exit:
            self._application_close_in_progress = True
        if getattr(self, "_close_in_progress", False):
            return
        self._close_in_progress = True
        pending = getattr(self, "_pending_raw_audio_csv_recording", None)
        if pending is not None:
            self._release_raw_audio_csv_recording(pending)

    def _raw_audio_csv_local_close_ready(self):
        paths = getattr(self, "_raw_audio_csv_recording_paths", set())
        paths.intersection_update(path for path in tuple(paths)
                                  if self.raw_audio_csv_service.paths_busy((path,)))
        return not (self._owned_raw_audio_csv_tasks or paths
                    or getattr(self, "_recording_process_contexts", None)
                    or (getattr(self, "_owns_recording_bridge", False)
                        and not getattr(self, "_sequence_recording_shutdown_reported", False)))

    def _wait_for_raw_audio_csv_close(self):
        """A Qt timer observes in-memory leases, never waits for worker I/O."""
        owned = getattr(self, "_owns_raw_audio_csv_service", False)
        if owned and not getattr(self, "_raw_audio_csv_shutdown_started", False):
            self._raw_audio_csv_shutdown_started = True
            self.raw_audio_csv_bridge.begin_shutdown()
        if (getattr(self, "_owns_recording_bridge", False)
                and not getattr(self, "_sequence_recording_shutdown_started", False)):
            self._sequence_recording_shutdown_started = True
            self.recording_bridge.shutdown(self._finish_sequence_recording_shutdown)
        if self._raw_audio_csv_local_close_ready() and (
                not owned or self.raw_audio_csv_bridge.service_closed):
            return False
        timer = getattr(self, "_raw_audio_csv_close_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(50)
            timer.timeout.connect(self._resume_raw_audio_csv_close)
            self._raw_audio_csv_close_timer = timer
        timer.start()
        self._show_raw_audio_csv_close_status()
        return True

    def _show_raw_audio_csv_close_status(self):
        count = len(self._owned_raw_audio_csv_tasks)
        text = f"正在保存/压缩原始 CSV（剩余 {count} 份），等待资源释放后关闭…"
        if not hasattr(self, "_raw_audio_csv_saved_title"):
            self._raw_audio_csv_saved_title = self.windowTitle()
        self.setWindowTitle(text)
        panel = getattr(self, "left_panel", None)
        if panel is not None:
            panel.set_current_stage(text, tone="pending")

    def _finish_sequence_recording_shutdown(self):
        self._sequence_recording_shutdown_reported = True
        if not self.recording_bridge.service.closed.is_set():
            self.default_logger.warning("Recording shutdown cleanup incomplete: %s",
                                        "\n".join(self.recording_bridge.service.diagnostics))
        # Retained recording path claims still gate CSV/local close readiness.
        self.close()

    def _resume_raw_audio_csv_close(self):
        self._show_raw_audio_csv_close_status()
        if self._raw_audio_csv_local_close_ready() and (
                not self._owns_raw_audio_csv_service or self.raw_audio_csv_bridge.service_closed):
            self._raw_audio_csv_close_timer.stop()
            self.close()

    def _schedule_raw_audio_csv_export(self, raw_channels):
        context = getattr(self, "_publishing_raw_audio_csv_context", None)
        if context is None:
            return False
        if not context.csv_enabled_snapshot:
            # WAV-only publication may immediately relabel the completed file.
            # Failed recording release still retains the claim via its readiness gate.
            SequenceWidgetRawCsvOpsMixin._release_raw_audio_csv_context(self, context)
            return False
        if context.csv_reservation is None:
            return False
        wav_path = context.request.path
        try:
            info = copy.deepcopy(context.recorded_signal_info)
            storage = storage_context_from_metadata(info.get("analysis_storage", {}))
            csv_path = str(build_raw_audio_csv_path(
                storage, os.path.splitext(os.path.basename(wav_path))[0]))
            channels = tuple(int(channel) for channel in raw_channels)
            if not wav_path or not channels or any(channel < 0 for channel in channels):
                raise ValueError("原始音频 CSV 路径或通道不可用")
            request = CsvExportRequest(uuid4().hex, context.request.request_id,
                wav_path, csv_path, channels,
                str(info.get("round_data_group_id", "")),
                str(info.get("round_data_record_key", "")))
            register = getattr(self, "_register_round_file", None)
            if callable(register):
                register(info, csv_path, is_raw_csv=True)
                register(info, str(raw_csv_zip_path(csv_path)), is_raw_csv=True)
        except (TypeError, ValueError, OSError) as error:
            self._release_raw_audio_csv_context(context)
            self._on_raw_audio_csv_export_failed(wav_path, str(error))
            return False
        # Routing and expected-file ownership precede dispatch, including fast workers.
        self._owned_raw_audio_csv_tasks.add(request.task_id)
        token = context.csv_reservation
        context.csv_reservation = None
        if context.csv_path_permit is None:
            status = self.raw_audio_csv_service.commit(token, request)
        else:
            status = self.raw_audio_csv_service.commit(
                token, request, recording_permit=context.csv_path_permit)
        if status != "accepted":
            self._owned_raw_audio_csv_tasks.discard(request.task_id)
            self._on_raw_audio_csv_export_failed(wav_path, f"CSV 提交失败：{status}")
            return False
        return True
