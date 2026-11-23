"""In-process test pause: coordinate existing capture and file owners."""
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
import math
import time

from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

from consts.running_consts import DEFAULT_DIR


@dataclass
class TestPauseContext:
    __test__ = False
    group_id: str
    condition_key: str
    recording_context: object
    recent_owner: object
    phase: str = "pausing"
    outcome: str = ""
    cleanup: object = None
    error: str = ""


def _delete_interrupted_wav(path, result):
    """CSV supervisor callback; never retain or access a widget."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as error:
        result.set_exception(error)
    else:
        result.set_result(None)


class SequenceWidgetPauseOpsMixin:
    def _init_test_pause(self):
        self._test_pause = None
        self._test_trigger_epoch = 0
        self._test_resume_after = 0.0
        self._test_pause_timer = QTimer(self)
        self._test_pause_timer.setInterval(100)
        self._test_pause_timer.timeout.connect(self._poll_test_pause)

    def _test_pause_blocks_start(self):
        return getattr(self, "_test_pause", None) is not None

    def _test_pause_is_busy(self):
        pause = getattr(self, "_test_pause", None)
        return pause is not None and pause.phase != "paused"

    def _test_pause_can_request(self):
        # Background analysis must not take over the manual next-capture button.
        # An already requested pause still waits for analysis in _poll_test_pause.
        return bool(
            getattr(self, "_manual_product_condition_group_id", "")
            and not getattr(self, "_round_reset_in_progress", False)
            and not getattr(self, "_round_reset_delete_failed", False)
            and not getattr(self, "_close_in_progress", False)
            and not getattr(self, "_application_close_in_progress", False)
            and not getattr(self, "_product_progress_choice_pending", False)
            and not getattr(self, "_product_test_program_config_dialog_open", False)
            and (getattr(self, "_record_workflow_busy", False)
                 or getattr(self, "player_status_flag", False)
                 or self._pause_uses_serial_trigger())
        )

    def _pause_uses_serial_trigger(self):
        return any(str(item.get("trigger_state") or "").strip()
                   for item in self.product_test_condition_configs)

    def _handle_test_pause_button(self):
        pause = getattr(self, "_test_pause", None)
        if pause is not None:
            if pause.phase == "paused":
                self._resume_test()
            elif pause.phase == "cleanup_failed":
                pause.error = ""
                pause.cleanup = None
                pause.phase = "pausing"
                self._test_pause_timer.start()
                self._poll_test_pause()
            return True
        if self._test_pause_can_request():
            self._request_test_pause()
            return True
        return False

    def _request_test_pause(self):
        if self._test_pause is not None or not self._test_pause_can_request():
            return
        session = getattr(self, "_recording_process_session", None)
        context = self._recording_context_for_session(session) if session is not None else None
        key = self._get_active_product_condition_key()
        if not key:
            conditions = self._product_condition_sequence()
            index = self._manual_product_condition_index
            if 0 <= index < len(conditions):
                key = self._product_condition_runtime_key(conditions[index], index)
        pause = TestPauseContext(
            self._manual_product_condition_group_id,
            key, context,
            self._recent_session_owner_snapshot())
        self._test_pause = pause
        self._test_trigger_epoch += 1
        self._queued_directional_trigger = ""
        self._pending_serial_trigger_direction = ""
        self._serial_product_pending_close_frame = ""
        self._serial_trigger_delay_timer.stop()
        self._barcode_debounce_timer.stop()
        # Do not mutate context.cancelled until the service chooses a terminal.
        # A completed result may already be queued for Qt delivery.
        if context is not None and not context.processing_notified:
            session.cancel()
        self._test_pause_timer.start()
        self._apply_test_pause_presentation()
        self._poll_test_pause()

    def _handle_test_pause_terminal(self, session, kind, error=""):
        pause = getattr(self, "_test_pause", None)
        if (pause is None or pause.recording_context is None
                or pause.recording_context.session is not session):
            return False
        context = pause.recording_context
        pause.outcome = kind
        context.final = True
        context.cancelled = kind == "cancelled"
        context.failed = kind == "failed"
        context.cleanup_owned = True
        context.preview_enabled = False
        context.accepted_audio = context.validated_audio = None
        self._end_streaming_waveform_session()
        if error:
            self._fail_test_pause(error)
        return True

    def _pause_prepared_recording(self):
        """Roll back startup if processEvents delivered pause before capture."""
        pending = getattr(self, "_pending_raw_audio_csv_recording", None)
        if pending is not None:
            self._release_raw_audio_csv_recording(pending)
        self._finalize_recording_channel_selection()
        self.player_status_flag = self._record_workflow_busy = False
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_latched_frame = ""
        self.left_panel.result_panel.pause_condition_capture(self._test_pause.condition_key)
        self._poll_test_pause()

    def _fail_test_pause(self, error):
        pause = self._test_pause
        pause.phase, pause.error = "cleanup_failed", str(error)
        self._test_pause_timer.stop()
        self.default_logger.error("Test pause incomplete: %s", error)
        self._apply_test_pause_presentation()

    def _handle_test_pause_business_error(self, reason):
        if not self._test_pause_blocks_start():
            return False
        # Complete-file publication can fail at the database boundary. Preserve
        # its WAV and earlier results, but do not claim this step completed.
        self.player_status_flag = False
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._sync_recording_workflow_busy()
        self._fail_test_pause(reason)
        return True

    def _poll_test_pause(self):
        pause = self._test_pause
        if pause is None or pause.phase != "pausing":
            return
        context = pause.recording_context
        if context is not None:
            session = context.session
            if session.release_error:
                self._fail_test_pause(session.release_error)
                return
            if pause.outcome in ("cancelled", "failed"):
                if not session.released.is_set():
                    return
                # Also handles release delivered before this timer tick.
                self._drop_recording_context(context)
                if pause.cleanup is None:
                    result = Future()
                    path = context.request.path
                    service = self.recording_bridge.service
                    queued = self.raw_audio_csv_service.defer_mutation(
                        (path,), lambda: _delete_interrupted_wav(path, result),
                        ready=lambda: not service.is_path_leased(path))
                    if not queued:
                        self._fail_test_pause("未能安排中断录音清理，请重试。")
                        return
                    pause.cleanup = result
                if not pause.cleanup.done():
                    return
                error = pause.cleanup.exception()
                if error is not None:
                    self._fail_test_pause(error)
                    return
                if self.raw_audio_csv_service.paths_busy((context.request.path,)):
                    return
                self._finish_interrupted_capture(pause)
            elif not context.publication_delivered:
                # accepted/released callbacks must publish complete recordings.
                return
        if (self._recording_contexts() or self._analysis_has_pending_tasks()
                or self.player_status_flag or self._record_workflow_busy
                or self.raw_audio_csv_service.snapshot().outstanding
                or getattr(self, "_recording_publication_in_progress", False)):
            self._apply_test_pause_presentation()
            return
        if not self._manual_product_condition_group_id:
            self._clear_test_pause()
            self.update_player_btn_is_paused()
            return
        pause.phase = "paused"
        self._test_pause_timer.stop()
        self._apply_test_pause_presentation()
        self._refresh_round_reset_button()

    def _finish_interrupted_capture(self, pause):
        context = pause.recording_context
        self._discard_recent_session_if_owned(pause.recent_owner)
        self._clear_unpublished_recording_source()
        self._finalize_recording_channel_selection()
        self.streaming_processor = None
        self._recording_process_audio = self._recording_process_validated_audio = None
        self._recording_process_final = self._recording_process_cancelled = True
        self._recording_ve_device = self._recording_wav_calibration_metadata = None
        self._product_recording_retry = self._recording_failure_recovery = None
        self.player_status_flag = False
        self._record_workflow_busy = False
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_latched_frame = ""
        panel = self.left_panel.result_panel
        panel.pause_condition_capture(context.condition_key)
        workspace = getattr(self, "channel_workspace", None)
        if workspace is not None:
            workspace.clear_direction(context.condition_key)
        pause.recording_context = None

    def _resume_test(self):
        pause = self._test_pause
        if pause is None or pause.phase != "paused":
            return
        if pause.group_id != self._manual_product_condition_group_id:
            self._clear_test_pause()
            self.update_player_btn_is_paused()
            return
        serial = self._pause_uses_serial_trigger()
        if serial and not getattr(self, "_serial_trigger_runtime_status", {}).get("connected", False):
            QMessageBox.information(self, "暂不能继续", "串口尚未连接，请恢复连接后再继续测试。")
            return
        # Resource admission must be checked without our start gate.
        self._test_pause = None
        if not self._can_prepare_recording_workflow():
            self._test_pause = pause
            QMessageBox.information(self, "暂不能继续", "设备或保存任务仍在占用资源，请稍后重试。")
            return
        self._test_trigger_epoch += 1
        self._test_resume_after = time.monotonic()
        if serial:
            self._serial_product_latched_frame = ""
            self.left_panel.set_current_stage("等待新的工况状态码", tone="pending")
            self.update_player_btn_is_paused()
            return
        before = getattr(self, "_recording_process_session", None)
        try:
            self.on_clicked_player_btn()
        finally:
            if (getattr(self, "_recording_process_session", None) is before
                    and self._test_pause is None
                    and self._manual_product_condition_group_id == pause.group_id):
                self._test_pause = pause
                self._apply_test_pause_presentation()

    def _clear_test_pause(self):
        self._test_pause_timer.stop()
        self._test_pause = None
        self._test_trigger_epoch += 1

    def _test_pause_accepts_serial_frame(self, payload):
        if self._test_pause_blocks_start():
            return False
        resumed_at = getattr(self, "_test_resume_after", 0.0)
        if not resumed_at:
            return True
        received = (payload or {}).get("received_monotonic")
        return (type(received) in (int, float) and math.isfinite(received)
                and received > resumed_at)

    def _apply_test_pause_presentation(self):
        pause = getattr(self, "_test_pause", None)
        if pause is None:
            if not self._test_pause_can_request():
                return False
            text, tooltip, enabled, icon = "", "暂停测试", True, "pause.png"
        elif pause.phase == "paused":
            text, enabled, icon = "已暂停", True, "play.png"
            tooltip = ("继续测试：等待新的有效状态码" if self._pause_uses_serial_trigger()
                       else "继续测试：从未完成档位重新完整录制")
        elif pause.phase == "cleanup_failed":
            text, tooltip, enabled, icon = "暂停未完成", "重试清理：" + pause.error, True, "pause.png"
        else:
            text, tooltip, enabled, icon = "正在暂停", "等待采集、保存及清理结束", False, "pause.png"
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/" + icon))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setToolTip(tooltip)
        self.player_btn.setEnabled(enabled)
        if pause is not None:
            self.data_btn.setEnabled(False)
            self.replayer_btn.setEnabled(False)
            self.left_panel.set_current_stage(text, tone="pending")
            workspace = getattr(self, "channel_workspace", None)
            if workspace is not None:
                workspace.set_condition_context(pause.condition_key, status=text)
        return True

    def _test_pause_allows_close(self):
        pause = getattr(self, "_test_pause", None)
        if pause is not None and pause.phase == "pausing":
            QMessageBox.information(self, "正在暂停", "请等待中断录音清理完成后再退出。")
            return False
        return True
