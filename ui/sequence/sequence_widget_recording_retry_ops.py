"""Condition-scoped recording recovery; all state belongs to the exact UI owner."""
from dataclasses import dataclass
import math
import os
import time

from PyQt5.QtWidgets import QMessageBox


@dataclass
class _ProductRecordingRetry:
    group_id: str
    condition_key: str
    condition_config: object
    index: int
    port_index: int
    workflow_token: object
    alias_session: object
    recent_owner: tuple
    context: object = None
    modal_pending: bool = True
    ready: bool = False
    received_after: float = 0.0


class SequenceWidgetRecordingRetryOpsMixin:
    def _capture_product_recording_retry(self, context=None):
        group_id = getattr(self, "_manual_product_condition_group_id", "")
        key = getattr(self, "_active_product_condition_key", "")
        if (not group_id or not key
                or key in getattr(self, "_manual_product_condition_completed_keys", set())
                or any(bool(getattr(self, name, False)) for name in (
                    "_recording_closed", "_closing", "_shutdown_started", "_close_in_progress",
                    "_recording_cleanup_in_progress", "_streaming_cleanup_in_progress",
                    "_round_reset_in_progress", "_serial_product_error_dialog_open"))):
            return None
        if context is not None and (
                context.cancelled or context.cleanup_owned or context.publication_started
                or not self._recording_context_owns_active_workflow(context)):
            return None
        retry = _ProductRecordingRetry(
            group_id, key, self._active_product_condition_config,
            self._manual_product_condition_index,
            getattr(self, "_serial_product_port_index", 0),
            getattr(self, "_recording_workflow_token", None),
            getattr(self, "_recording_process_session", None),
            self._recent_session_owner_snapshot(), context)
        self._product_recording_retry = retry
        return retry

    def _current_product_recording_retry(self):
        retry = getattr(self, "_product_recording_retry", None)
        if retry is None:
            return None
        context = retry.context
        bridge = getattr(self, "recording_bridge", None)
        if (retry.group_id != getattr(self, "_manual_product_condition_group_id", "")
                or retry.condition_key != getattr(self, "_active_product_condition_key", "")
                or retry.condition_config is not getattr(self, "_active_product_condition_config", None)
                or retry.index != getattr(self, "_manual_product_condition_index", 0)
                or retry.port_index != getattr(self, "_serial_product_port_index", 0)
                or retry.workflow_token is not getattr(self, "_recording_workflow_token", None)
                or retry.alias_session is not getattr(self, "_recording_process_session", None)
                or (context is not None and (context.cancelled or context.cleanup_owned))
                or getattr(bridge, "_shutdown_requested", False)
                or any(bool(getattr(self, name, False)) for name in (
                    "_recording_closed", "_closing", "_shutdown_started", "_close_in_progress",
                    "_recording_cleanup_in_progress", "_streaming_cleanup_in_progress",
                    "_round_reset_in_progress"))):
            self._product_recording_retry = None
            return None
        return retry

    def _observe_product_retry_readiness(self, session=None):
        retry = self._current_product_recording_retry()
        if retry is None or retry.modal_pending:
            return False
        if session is not None and (retry.context is None or retry.context.session is not session):
            return False
        if not self._can_prepare_recording_workflow():
            retry.ready = False
            return False
        if not retry.ready:
            # No poll/automatic retry: conservatively discard everything received
            # before the GUI can prove that all recording/CSV owners have left.
            retry.received_after = time.monotonic()
            retry.ready = True
        return True

    def _product_retry_accepts_frame(self, payload, received_frame):
        retry = self._current_product_recording_retry()
        if retry is None:
            return True
        if not self._observe_product_retry_readiness():
            return False
        received = (payload or {}).get("received_monotonic")
        if (isinstance(received, bool) or not isinstance(received, (int, float))
                or not math.isfinite(received) or received <= retry.received_after):
            return False
        from base.hardware_trigger.serial_full_frame_matcher import normalize_frame_candidates
        frame = retry.condition_config.get("trigger_state", "")
        return bool(frame) and received_frame == normalize_frame_candidates([frame])[0]

    def _recover_current_product_recording(self, reason):
        retry = self._current_product_recording_retry()
        if retry is None:
            retry = self._capture_product_recording_retry()
        if retry is None:
            return False
        # Process-service ownership includes file deletion after resource release.
        # Legacy WAVs can be removed only when neither recording nor CSV owns them.
        path = str(getattr(self, "recorded_path", "") or "")
        session = retry.alias_session
        process_owned = session is not None and path and os.path.normcase(os.path.abspath(path)) == os.path.normcase(session.request.path)
        if path and not process_owned and not self._recording_path_is_leased(path):
            csv_service = getattr(self, "raw_audio_csv_service", None)
            permit = csv_service.try_acquire_mutation((path,)) if csv_service is not None else None
            try:
                if (csv_service is None or permit is not None) and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError as error:
                        self.default_logger.warning(f"remove_invalid_recording_failed path={path} err={error}")
            finally:
                if permit is not None:
                    csv_service.release_mutation(permit)
        self._discard_recent_session_if_owned(retry.recent_owner)
        self._clear_unpublished_recording_source()
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_pending_close_frame = ""
        self._queued_directional_trigger = ""
        self._pending_serial_trigger_direction = ""
        self._pending_recent_session_append = False
        self.player_status_flag = False
        self.clicked_player_flag = False
        self._sync_recording_workflow_busy()
        self._set_active_product_condition_stage("测试异常", tone="ng")
        self._serial_product_error_dialog_open = True
        self.update_player_btn_is_paused()
        try:
            QMessageBox.warning(self, "录音异常", reason)
        finally:
            if self._current_product_recording_retry() is retry:
                self._serial_product_error_dialog_open = False
                retry.modal_pending = False
                retry.received_after = time.monotonic()
                self._observe_product_retry_readiness()
                self.update_player_btn_is_paused()
            elif getattr(self, "_product_recording_retry", None) is None:
                self._serial_product_error_dialog_open = False
        return True
