"""Main-recording adapter: request snapshots, envelopes and accepted results."""
import os
import time
from uuid import uuid4

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from base.play_and_record import resolve_startup_trim_samples
from base.recording_process_protocol import RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from base.recording_settings import merge_audio_validation_thresholds
from ui.recording_service_bridge import RecordingProcessorFacade, RecordingServiceBridge


class SequenceWidgetRecordingProcessOpsMixin:
    def _get_recording_bridge(self):
        bridge = getattr(self, "recording_bridge", None)
        if bridge is None:
            # Isolated window callers own this instance. Construction opens no device.
            bridge = RecordingServiceBridge(RecordingService())
            self.recording_bridge = bridge
            self._owns_recording_bridge = True
        return bridge

    def _start_process_recording(self, recorded_dict, sample_rate, *, tcp_completion_address=None):
        detail = self._resolve_recording_acq_detail()
        channels = tuple(self._recording_input_channels)
        monitor = {
            "enabled": bool(recorded_dict.get("monitor_playback", False)),
            "device": recorded_dict.get("output_device"),
            "channels": tuple(recorded_dict.get("output_channels") or ()),
            "gain_db": float(recorded_dict.get("monitor_gain_db", 0)),
            "mute_leading_samples": int(recorded_dict.get("monitor_mute_leading_samples", 0)),
            "fade_in_samples": int(recorded_dict.get("monitor_fade_in_samples", 0)),
        }
        request = RecordingRequest(
            uuid4().hex, "main", int(sample_rate), int(recorded_dict["num_frames"]),
            channels, recorded_dict["device"], os.path.abspath(self.recorded_path),
            bool(self._should_use_streaming_recording()),
            resolve_startup_trim_samples(detail, sample_rate), monitor,
            getattr(self, "_recording_wav_calibration_metadata", None),
            merge_audio_validation_thresholds(detail))
        self._end_streaming_waveform_session()
        self._recording_process_preview_enabled = request.effective_streaming
        self._recording_process_sequence = 0
        self._recording_process_final = False
        self._recording_process_cancelled = False
        self._recording_process_cleanup_owned = False
        self._recording_release_warned = False
        self._recording_process_audio = None
        self._recording_process_windows = None
        self._recording_process_direction = self._resolve_active_recording_waveform_direction(fallback="")
        # The placeholder and UI token exist before any callback can be delivered.
        self._begin_recent_session_for_current_run()
        self._recording_process_id = request.request_id
        self._recording_process_tcp_completion = (request.request_id, tcp_completion_address)
        callbacks = RecordingCallbacks(
            started=self._on_process_recording_started,
            preview=self._on_process_recording_preview,
            result_ready=self._on_process_recording_result,
            accepted=self._on_process_recording_accepted,
            failed=self._on_process_recording_failed,
            cancelled=self._on_process_recording_cancelled,
            released=self._on_process_recording_released,
            release_failed=self._on_process_recording_release_failed)
        try:
            session = self._get_recording_bridge().start(request, callbacks)
        except (RuntimeError, ValueError, TypeError):
            self._recording_process_id = None
            self._recording_process_tcp_completion = None
            discard = getattr(self, "_discard_current_recent_session", None)
            if callable(discard):
                discard()
            raise
        self._recording_process_session = session
        self.streaming_processor = RecordingProcessorFacade(session)
        self.streaming_mode = "record_only"
        self.streaming_stimulus_data = None

    def _is_current_recording_process(self, session):
        return getattr(self, "_recording_process_id", None) == session.request.request_id

    def _on_process_recording_started(self, session):
        if self._is_current_recording_process(session):
            self._set_active_product_condition_stage("采集中")
            self.default_logger.info(f"Recording started request={session.request.request_id} pid={session.worker_pid}")

    def _on_process_recording_preview(self, session, preview):
        if (not self._is_current_recording_process(session)
                or self._recording_process_final or not self._recording_process_preview_enabled
                or preview.generation != session.generation
                or preview.sequence <= self._recording_process_sequence):
            return
        self._recording_process_sequence = preview.sequence
        try:
            if preview.channels != session.request.channels:
                raise ValueError("preview physical channels differ from request")
            windows = self._validate_final_waveform_workspace(session.request.channels)
            for window, waveform in zip(windows, preview.waveforms):
                # A snapshot replaces the complete cumulative display. Never append.
                window.set_data(waveform.time, waveform.amplitude)
        except Exception as error:
            # Plot implementations are an external presentation boundary. A broken
            # widget disables live display only; capture/final validation continue.
            self._recording_process_preview_enabled = False
            self.default_logger.warning(f"Recording live preview disabled: {error}")

    def _on_process_recording_result(self, session, audio):
        if not self._is_current_recording_process(session) or self._recording_process_cancelled:
            session.reject_result("recording view was cancelled or replaced")
            return
        self._recording_process_final = True
        try:
            request, descriptor = session.request, audio.descriptor
            multi = self._normalize_final_recording_array(audio.multi, request.channels)
            if (descriptor.channels != request.channels or descriptor.path != request.path
                    or descriptor.sample_rate != request.sample_rate
                    or len(multi) != descriptor.final_frames
                    or not np.array_equal(audio.mono, multi.mean(axis=1), equal_nan=True)):
                raise ValueError("final recording arrays do not match the request/result contract")
            self._recording_process_windows = self._validate_final_waveform_workspace(request.channels)
        except (TypeError, ValueError, OverflowError) as error:
            session.reject_result(str(error))
            return
        session.accept_result()

    def _on_process_recording_accepted(self, session, audio):
        if self._is_current_recording_process(session) and not self._recording_process_cancelled:
            # Business completion can relabel/move or start the next round. Wait
            # for released as well as accepted before entering that code.
            self._recording_process_audio = audio
            self._recording_release_deadline = time.monotonic() + 5
            QTimer.singleShot(0, lambda: self._check_accepted_recording_release(session))

    def _check_accepted_recording_release(self, session):
        if (not self._is_current_recording_process(session)
                or self._recording_process_audio is None or self._recording_process_cancelled
                or session.released.is_set()):
            return
        if session.release_error is not None or time.monotonic() >= self._recording_release_deadline:
            # Accepted audio remains successful even if optional cleanup cannot
            # release its path. All file mutations still consult the service lease.
            self._on_process_recording_release_failed(
                session, session.release_error or "等待文件资源释放超时")
        else:
            QTimer.singleShot(50, lambda: self._check_accepted_recording_release(session))

    def _on_process_recording_released(self, session):
        self._publish_process_recording(session)

    def _on_process_recording_release_failed(self, session, error):
        if (not self._is_current_recording_process(session)
                or self._recording_process_cancelled
                or self._recording_process_cleanup_owned
                or self._recording_release_warned):
            return
        # Claim before the modal warning can dispatch close/abort/new-session
        # events. Accepted audio stays valid and the exact file lease stays held.
        self._recording_release_warned = True
        self.default_logger.warning(
            f"录音文件资源未释放: {session.request.path}; {error}")
        QMessageBox.warning(self, "录音资源未释放",
            "录音文件资源未能释放，请勿重录或移动此文件。此问题不会改变已接纳的录音结果。\n"
            + session.request.path + "\n" + str(error))
        if not self._recording_process_cancelled:
            self._publish_process_recording(session)

    def _publish_process_recording(self, session):
        if not self._is_current_recording_process(session):
            return
        audio = self._recording_process_audio
        if audio is not None and session.state == "completed":
            self._recording_process_audio = None
            for warning in audio.descriptor.warnings:
                self.default_logger.warning(f"Recording {session.request.request_id}: {warning}")
            try:
                succeeded = self._on_streaming_complete(
                    recorded_mono=audio.mono, recorded_multi=audio.multi,
                    sample_rate=audio.descriptor.sample_rate,
                    completion_source="process", prefinalized=True,
                    final_waveform_windows=self._recording_process_windows)
            finally:
                if self._is_current_recording_process(session):
                    self._finalize_recording_channel_selection()
            if succeeded is True:
                self._notify_process_recording_finished(session)

    def _notify_process_recording_finished(self, session):
        if (not self._is_current_recording_process(session)
                or self._recording_process_cancelled
                or self._recording_process_cleanup_owned):
            return
        completion = self._recording_process_tcp_completion
        if completion is None or completion[0] != session.request.request_id:
            return
        # Claim before network I/O, including a send that raises after delivery.
        self._recording_process_tcp_completion = None
        if completion[1] is not None:
            self._send_recording_tcp_finish(completion[1])

    def _on_process_recording_failed(self, session, failure):
        if not self._is_current_recording_process(session):
            return
        self.default_logger.error(
            f"Recording failed request={session.request.request_id} "
            f"stage={failure.stage} path={session.request.path}: {failure.message}")
        if getattr(self, "_recording_process_cleanup_owned", False):
            # Abort/close already chose the controls, history and stage. A
            # cancellation that fails while closing still belongs to that
            # caller; diagnose it without applying ordinary failure recovery.
            return
        self._recording_process_final = True
        self._recording_process_audio = None
        self.streaming_processor = None
        self._handle_invalid_recording(f"录音失败 ({failure.stage}): {failure.message}")

    def _on_process_recording_cancelled(self, session, descriptor):
        if (not self._is_current_recording_process(session)
                or getattr(self, "_recording_process_cleanup_owned", False)):
            return
        self._recording_process_final = True
        self._recording_process_audio = None
        self.streaming_processor = None
        handled = self._cleanup_failed_recording_initialization("录音已取消")
        if handled or not self._is_current_recording_process(session):
            return
        discard = getattr(self, "_discard_current_recent_session", None)
        if callable(discard):
            discard()

    def _cancel_process_recording(self):
        session = getattr(self, "_recording_process_session", None)
        if session is None:
            return
        self._recording_process_preview_enabled = False
        self._recording_process_final = True
        self._recording_process_audio = None
        self._recording_process_cancelled = True
        # Resource cleanup callers (serial abort, close, replacement) decide
        # their own controls/history disposition. A late cancellation cannot
        # run ordinary completion cleanup over that decision.
        self._recording_process_cleanup_owned = True
        session.cancel()

    def _recording_path_is_leased(self, path):
        bridge = getattr(self, "recording_bridge", None)
        return bridge is not None and bridge.service.is_path_leased(path)
