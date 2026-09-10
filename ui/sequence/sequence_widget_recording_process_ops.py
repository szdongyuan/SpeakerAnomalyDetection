"""Main-recording adapter: request snapshots, envelopes and accepted results."""
import os
import time
from uuid import uuid4

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from base.play_and_record import resolve_startup_trim_samples
from base.recording_preview_config import resolve_recording_preview_time_mode
from base.recording_process_protocol import RecordingFailure, RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from base.recording_settings import merge_audio_validation_thresholds
from base.ve3668n_input import validate_sample_rate
from base.ve3668n_wav_metadata import validate_ve_wav_metadata
from consts.ve3668n_consts import VE_BACKEND
from ui.recording_service_bridge import RecordingProcessorFacade, RecordingServiceBridge
from ui.sequence.recording_process_context import RecordingProcessContext



class SequenceWidgetRecordingProcessOpsMixin:
    def _recording_contexts(self):
        contexts = getattr(self, "_recording_process_contexts", None)
        if not isinstance(contexts, dict):
            contexts = {}
            self._recording_process_contexts = contexts
        return contexts

    def _recent_session_owner_snapshot(self):
        session_id = getattr(self, "_current_recent_session_id", None)
        records = getattr(self, "recent_test_session_by_id", None)
        record = (records.get(session_id)
                  if session_id is not None and isinstance(records, dict)
                  else None)
        return session_id, record

    @staticmethod
    def _recent_session_owner_changed(previous, current):
        return previous[0] != current[0] or previous[1] is not current[1]

    def _discard_recent_session_if_owned(self, owner):
        if owner is None:
            return False
        current = self._recent_session_owner_snapshot()
        if current[0] != owner[0] or current[1] is not owner[1]:
            return False
        discard = getattr(self, "_discard_current_recent_session", None)
        if not callable(discard):
            return False
        discard()
        return True

    def _cancel_failed_recording_start_sessions(self, context, *sessions):
        """Best-effort rollback for sessions exposed by a failed start contract."""
        context.cleanup_owned = True
        distinct = []
        for session in sessions:
            if session is None or any(session is item for item in distinct):
                continue
            distinct.append(session)
            cancel = getattr(session, "cancel", None)
            if not callable(cancel):
                continue
            try:
                cancel()
            except Exception as cancel_error:
                # Session cancellation is an external lifecycle boundary. A
                # failure is diagnosed, but cannot hide the start/install error
                # or prevent another possibly-started session from cancellation.
                self.default_logger.error(
                    "Recording start rollback cancel failed "
                    f"request={context.request.request_id}: {cancel_error}")

    def _recording_context_for_session(self, session):
        contexts = getattr(self, "_recording_process_contexts", None)
        if not isinstance(contexts, dict):
            return None
        request = getattr(session, "request", None)
        request_id = getattr(request, "request_id", None)
        context = contexts.get(request_id)
        if context is None:
            return None
        if context.session is session:
            return context
        if (context.session is None
                and context.session_binding_pending
                and context.request is request):
            # RecordingServiceBridge may emit `started` synchronously before
            # start() returns. The first exact-request callback owns the
            # provisional slot; later same-ID session objects cannot replace it.
            context.session = session
            return context
        return None

    def _is_active_recording_process(self, session):
        request = getattr(session, "request", None)
        request_id = getattr(request, "request_id", None)
        active_id = getattr(self, "_active_recording_process_id", None)
        if active_id is None:
            active_id = getattr(self, "_recording_process_id", None)
        contexts = getattr(self, "_recording_process_contexts", None)
        if isinstance(contexts, dict) and request_id in contexts:
            return (active_id == request_id
                    and SequenceWidgetRecordingProcessOpsMixin
                    ._recording_context_for_session(self, session)
                    is contexts[request_id])
        return active_id == request_id




    def _can_start_recording_workflow(self):
        """Combine local non-capture blockers with the service's atomic admission state."""
        if any(bool(getattr(self, name, False)) for name in (
                "_recording_closed", "_closing", "_shutdown_started", "_close_in_progress",
                "_test_metadata_validation_open", "_product_test_program_config_dialog_open",
                "_serial_product_error_dialog_open", "_recording_cleanup_in_progress",
                "_streaming_cleanup_in_progress")):
            return False
        controllers = [getattr(self, "_condition_playback_controller", None)]
        recent_panel = getattr(self, "recent_session_panel", None)
        controllers.append(
            getattr(recent_panel, "playback_controller", None)
            if recent_panel is not None else None)
        for controller in controllers:
            if controller is not None and controller.is_audio_playing():
                return False
        if (getattr(self, "_recording_publication_in_progress", False)
                or getattr(self, "_recording_process_contexts", None)
                or getattr(self, "_record_workflow_busy", False)
                or getattr(self, "player_status_flag", False)):
            return False
        bridge = getattr(self, "recording_bridge", None)
        return bridge is None or bool(getattr(
            bridge.service, "can_start_recording", not getattr(bridge.service, "busy", False)))

    def _can_start_calibration_workflow(self):
        return self._can_start_recording_workflow()

    def _recording_context_owns_active_workflow(self, context):
        if context is None:
            return True
        if getattr(self, "_active_recording_process_id", None) != context.request.request_id:
            return False
        token = getattr(context, "workflow_token", None)
        return token is None or getattr(self, "_recording_workflow_token", None) is token

    def _sync_recording_workflow_busy(self):
        self._record_workflow_busy = bool(getattr(self, "_recording_process_contexts", {}) or {})

    def _drop_recording_context(self, context):
        contexts = self._recording_contexts()
        request_id = context.request.request_id
        if contexts.get(request_id) is not context:
            return False
        contexts.pop(request_id)
        if getattr(self, "_active_recording_process_id", None) == request_id:
            self._active_recording_process_id = None
        self._sync_recording_workflow_busy()
        return True

    def _get_recording_bridge(self):
        bridge = getattr(self, "recording_bridge", None)
        if bridge is None:
            # Isolated window callers own this instance. Construction opens no device.
            bridge = RecordingServiceBridge(RecordingService())
            self.recording_bridge = bridge
            self._owns_recording_bridge = True
        return bridge

    def _start_process_recording(self, recorded_dict, sample_rate, *, tcp_completion_address=None):
        if (any(not context.cancelled and not context.cleanup_owned
                for context in self._recording_contexts().values())
                or getattr(self, "_recording_publication_in_progress", False)):
            raise RuntimeError("previous recording publication has not entered the analysis queue")
        detail = self._resolve_recording_acq_detail()
        preview_time_mode = resolve_recording_preview_time_mode(detail)
        if recorded_dict["device"].get("backend") == VE_BACKEND:
            request_rate = validate_sample_rate(sample_rate)
        else:
            request_rate = int(sample_rate)
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
            uuid4().hex, "main", request_rate, int(recorded_dict["num_frames"]),
            channels, recorded_dict["device"], os.path.abspath(self.recorded_path),
            bool(self._should_use_streaming_recording()),
            resolve_startup_trim_samples(detail, sample_rate), monitor,
            getattr(self, "_recording_wav_calibration_metadata", None),
            merge_audio_validation_thresholds(detail),
            preview_time_mode)
        previous_recent_owner = self._recent_session_owner_snapshot()
        attempt_recent_owner = None
        recent_placeholder_attempted = False
        try:
            self._end_streaming_waveform_session()
            direction = self._resolve_active_recording_waveform_direction(fallback="")
            # The placeholder and UI token exist before any callback can be delivered.
            recent_placeholder_attempted = True
            self._begin_recent_session_for_current_run()
            candidate_recent_owner = self._recent_session_owner_snapshot()
            if self._recent_session_owner_changed(
                    previous_recent_owner, candidate_recent_owner):
                attempt_recent_owner = candidate_recent_owner
            context = RecordingProcessContext(
                request=request,
                direction=direction,
                tcp_completion=(request.request_id, tcp_completion_address),
                preview_enabled=request.effective_streaming,
                session_binding_pending=True,
                recent_session_id=str(
                    getattr(self, "_current_recent_session_id", "") or ""),
                recorded_signal_info=dict(
                    getattr(self, "recorded_signal_info", {}) or {}),
                workflow_token=getattr(self, "_recording_workflow_token", None),
            )
            validate_workspace = getattr(
                self, "_validate_final_waveform_workspace", None)
            if callable(validate_workspace):
                try:
                    context.final_windows = validate_workspace(request.channels)
                except (TypeError, ValueError, OverflowError):
                    context.final_windows = None
        except Exception:
            if recent_placeholder_attempted and attempt_recent_owner is None:
                candidate_recent_owner = self._recent_session_owner_snapshot()
                if self._recent_session_owner_changed(
                        previous_recent_owner, candidate_recent_owner):
                    attempt_recent_owner = candidate_recent_owner
            self._discard_recent_session_if_owned(attempt_recent_owner)
            raise
        self._recording_contexts()[request.request_id] = context
        self._active_recording_process_id = request.request_id
        callbacks = RecordingCallbacks(
            started=self._on_process_recording_started,
            preview=self._on_process_recording_preview,
            result_ready=self._on_process_recording_result,
            accepted=self._on_process_recording_accepted,
            failed=self._on_process_recording_failed,
            cancelled=self._on_process_recording_cancelled,
            released=self._on_process_recording_released,
            release_failed=self._on_process_recording_release_failed)
        session = None
        try:
            session = self._get_recording_bridge().start(request, callbacks)
            if self._recording_contexts().get(request.request_id) is not context:
                # A synchronous terminal callback already performed exact-owner
                # cleanup. Do not reinstall aliases or a processor afterward.
                context.session_binding_pending = False
                return
            if context.session is None:
                context.session = session
            elif context.session is not session:
                raise RuntimeError(
                    "recording bridge returned a different session than its callback")
            context.session_binding_pending = False
            processor = RecordingProcessorFacade(session)
        except Exception:
            context.session_binding_pending = False
            self._cancel_failed_recording_start_sessions(
                context, context.session, session)
            self._drop_recording_context(context)
            current_process_id = getattr(self, "_recording_process_id", None)
            if current_process_id in (None, request.request_id):
                self._recording_process_id = None
                self._recording_process_tcp_completion = None
            self._discard_recent_session_if_owned(attempt_recent_owner)
            raise
        # Compatibility aliases describe the active capture only. Install them
        # only after the bridge's returned identity and registry owner agree.
        self._recording_process_preview_enabled = context.preview_enabled
        self._recording_process_sequence = 0
        self._recording_process_final = False
        self._recording_process_failed = False
        self._recording_process_cancelled = False
        self._recording_process_cleanup_owned = False
        self._recording_release_warned = False
        self._recording_process_audio = None
        self._recording_process_validated_audio = None
        self._recording_process_request = request
        self._recording_process_windows = context.final_windows
        self._recording_process_direction = direction
        self._recording_process_id = request.request_id
        self._recording_process_tcp_completion = context.tcp_completion
        self._recording_process_session = session
        context.processor = processor
        self.streaming_processor = context.processor
        self.streaming_mode = "record_only"
        self.streaming_stimulus_data = None

    def _is_current_recording_process(self, session):
        contexts = getattr(self, "_recording_process_contexts", None)
        request = getattr(session, "request", None)
        request_id = getattr(request, "request_id", None)
        if isinstance(contexts, dict) and request_id in contexts:
            return (SequenceWidgetRecordingProcessOpsMixin
                    ._recording_context_for_session(self, session)
                    is contexts[request_id])
        # Compatibility for isolated legacy tests that seed only the old ID.
        return getattr(self, "_recording_process_id", None) == request_id

    def _on_process_recording_started(self, session):
        if self._is_active_recording_process(session):
            self._set_active_product_condition_stage(
                "采集中",
                update_task_stage=False,
            )
            has_pending_analysis = getattr(
                self,
                "_analysis_has_pending_tasks",
                None,
            )
            if not (
                callable(has_pending_analysis)
                and has_pending_analysis()
            ):
                left_panel = getattr(self, "left_panel", None)
                clear_stage = getattr(left_panel, "clear_current_stage", None)
                if callable(clear_stage):
                    clear_stage()
            self.default_logger.info(f"Recording started request={session.request.request_id} pid={session.worker_pid}")

    def _on_process_recording_preview(self, session, preview):
        context = self._recording_context_for_session(session)
        final = (context.final if context is not None
                 else getattr(self, "_recording_process_final", False))
        preview_enabled = (context.preview_enabled if context is not None
                           else getattr(self, "_recording_process_preview_enabled", False))
        sequence = (context.sequence if context is not None
                    else getattr(self, "_recording_process_sequence", 0))
        if (not self._is_active_recording_process(session)
                or final or not preview_enabled
                or preview.generation != session.generation
                or preview.sequence <= sequence):
            return
        if context is not None:
            context.sequence = preview.sequence
        self._recording_process_sequence = preview.sequence
        try:
            if preview.time_mode != session.request.preview_time_mode:
                raise ValueError("recording preview time mode differs from request")
            if preview.channels != session.request.channels:
                raise ValueError("preview physical channels differ from request")
            windows = self._project_live_waveforms_to_workspace(
                session.request.channels,
                preview.waveforms,
                preview.time_mode,
            )
            ve_recording = getattr(session.request, "device", {}).get("backend") == VE_BACKEND
            if ve_recording:
                self._set_recording_voltage_tooltips(windows, session.request.calibration_metadata)
            else:
                clear_voltage_hint = getattr(self, "_clear_recording_voltage_tooltips", None)
                if callable(clear_voltage_hint):
                    clear_voltage_hint(windows)
        except Exception as error:
            # Plot implementations are an external presentation boundary. A broken
            # widget disables live display only; capture/final validation continue.
            if context is not None:
                context.preview_enabled = False
            self._recording_process_preview_enabled = False
            self.default_logger.warning(f"Recording live preview disabled: {error}")

    def _on_process_recording_result(self, session, audio):
        context = self._recording_context_for_session(session)
        if context is None or context.cancelled or context.failed:
            session.reject_result("recording view was cancelled or replaced")
            return
        context.final = True
        if self._is_active_recording_process(session):
            self._recording_process_final = True
        try:
            request, descriptor = session.request, audio.descriptor
            if getattr(request, "device", {}).get("backend") == VE_BACKEND:
                self._validate_ve_recording_result(session, audio, context=context)
            multi = self._normalize_final_recording_array(audio.multi, request.channels)
            if (descriptor.channels != request.channels or descriptor.path != request.path
                    or descriptor.sample_rate != request.sample_rate
                    or len(multi) != descriptor.final_frames
                    or not np.array_equal(audio.mono, multi.mean(axis=1), equal_nan=True)):
                raise ValueError("final recording arrays do not match the request/result contract")
            if context.final_windows is None:
                context.final_windows = self._validate_final_waveform_workspace(
                    request.channels)
        except (TypeError, ValueError, OverflowError) as error:
            context.failed = True
            session.reject_result(str(error))
            return
        context.validated_audio = audio
        if self._is_active_recording_process(session):
            self._recording_process_validated_audio = audio
        session.accept_result()

    def _validate_ve_recording_result(self, session, audio, *, context=None):
        """Admission check; never reopen a leased WAV or consult current stores."""
        request, descriptor = session.request, audio.descriptor
        context = context or self._recording_context_for_session(session)
        if context is None or request is not context.request:
            raise ValueError("VE result does not belong to the frozen recording request")
        if (not isinstance(audio.multi, np.ndarray) or not isinstance(audio.mono, np.ndarray)
                or audio.multi.dtype != np.float32 or audio.mono.dtype != np.float32
                or audio.multi.ndim != 2 or audio.mono.ndim != 1):
            raise ValueError("VE result must contain frame-first float32 voltage arrays")
        metadata = validate_ve_wav_metadata(request.calibration_metadata)
        expected_acquisition = {
            "model": request.device["model"], "machine_id": request.device["machine_id"],
            **request.device["input_config"],
        }
        trim = request.trim_samples if request.trim_samples < request.target_samples else 0
        if (metadata["acquisition"] != expected_acquisition
                or tuple(item["physical_input_channel"] for item in metadata["recorded_channels"]) != request.channels
                or descriptor.request_id != request.request_id
                or descriptor.purpose != "main"
                or any(type(value) is not int for value in (
                    descriptor.sample_rate, descriptor.raw_frames, descriptor.final_frames))
                or descriptor.raw_frames != request.target_samples
                or descriptor.final_frames != request.target_samples - trim
                or descriptor.metadata_appended is not True
                or descriptor.handles_released is not True
                or not np.all(np.isfinite(audio.multi))
                or not np.all(np.isfinite(audio.mono))):
            raise ValueError("VE final recording metadata/count/release contract mismatch")

    def _on_process_recording_accepted(self, session, audio):
        context = self._recording_context_for_session(session)
        if context is not None and not context.cancelled and not context.failed:
            if audio is not context.validated_audio:
                return
            # Business completion can relabel/move or start the next round. Wait
            # for released as well as accepted before entering that code.
            context.accepted_audio = audio
            context.release_deadline = time.monotonic() + 5
            if self._is_active_recording_process(session):
                self._recording_process_audio = audio
                self._recording_release_deadline = context.release_deadline
            QTimer.singleShot(0, lambda: self._check_accepted_recording_release(session))

    def _check_accepted_recording_release(self, session):
        context = self._recording_context_for_session(session)
        if context is None or context.accepted_audio is None or context.cancelled:
            return
        if session.released.is_set():
            self._publish_recording_context(context)
        else:
            deadline = context.release_deadline
            if self._is_active_recording_process(session):
                deadline = min(deadline, getattr(self, "_recording_release_deadline", deadline))
            if session.release_error is None and time.monotonic() < deadline:
                QTimer.singleShot(50, lambda: self._check_accepted_recording_release(session))
                return
            # Backend-specific release policy; file mutations always consult
            # the service lease, including after an optional legacy warning.
            self._on_process_recording_release_failed(
                session, session.release_error or "等待文件资源释放超时")

    def _on_process_recording_released(self, session):
        context = self._recording_context_for_session(session)
        if context is not None:
            if context.cleanup_owned or context.cancelled or context.failed:
                self._drop_recording_context(context)
            else:
                self._publish_recording_context(context)

    def _on_process_recording_release_failed(self, session, error):
        context = self._recording_context_for_session(session)
        if context is None:
            return
        if context.cleanup_owned or context.cancelled:
            self._drop_recording_context(context)
            return
        if context.release_warned:
            return
        # Claim before the modal warning can dispatch close/abort/new-session
        # events. The exact file lease stays held on either backend.
        context.release_warned = True
        self.default_logger.warning(
            f"录音文件资源未释放: {session.request.path}; {error}")
        if getattr(session.request, "device", {}).get("backend") == VE_BACKEND:
            self._on_process_recording_failed(session, RecordingFailure(
                session.request.request_id, "release", session.request.path,
                f"VE 录音资源未释放，未发布结果。请等待资源释放后重试。{error}",
                handles_released=False))
            return
        QMessageBox.warning(self, "录音资源未释放",
            "录音文件资源未能释放，请勿重录或移动此文件。此问题不会改变已接纳的录音结果。\n"
            + session.request.path + "\n" + str(error))
        if not context.cancelled:
            self._publish_recording_context(context)

    def _publish_process_recording(self, session):
        context = self._recording_context_for_session(session)
        if context is not None:
            return self._publish_recording_context(context)


    def _publish_recording_context(self, context):
        session = context.session
        if (session is None or context.cancelled or context.failed or context.cleanup_owned
                or context.publication_started
                or self._recording_context_for_session(session) is not context
                or not self._is_active_recording_process(session)):
            return
        audio = context.accepted_audio
        if audio is None or session.state != "completed":
            return
        released = session.released.is_set() and session.release_error is None
        # Soundcard temp-cleanup failure is optional in the original pipeline:
        # accepted audio remains publishable after its once-only warning. The
        # service keeps the exact file lease. VE requires confirmed release.
        optional_soundcard_cleanup = (
            context.request.device.get("backend") != VE_BACKEND and context.release_warned)
        if not released and not optional_soundcard_cleanup:
            return
        if session.request is not context.request or audio is not context.validated_audio:
            return
        # Claim before GUI observers can reenter. Admission remains held through
        # the original queue's snapshot of the current recording and config.
        context.publication_started = True
        context.accepted_audio = None
        self._recording_publication_in_progress = True
        self.recorded_path = context.request.path
        self.recorded_signal_info = dict(context.recorded_signal_info or self.recorded_signal_info)
        self._recording_input_channels = tuple(context.request.channels)
        self._active_input_channels = list(context.request.channels)
        self._recording_process_request = context.request
        self._recording_process_direction = context.direction
        try:
            for warning in audio.descriptor.warnings:
                self.default_logger.warning(f"Recording {session.request.request_id}: {warning}")
            succeeded = self._on_streaming_complete(
                recorded_mono=audio.mono, recorded_multi=audio.multi,
                sample_rate=audio.descriptor.sample_rate,
                completion_source="process", prefinalized=True,
                final_waveform_windows=context.final_windows)
            if succeeded is True:
                self._notify_process_recording_finished(session, context=context)
        finally:
            if self._is_active_recording_process(session):
                self._finalize_recording_channel_selection()
            context.publication_delivered = True
            self._drop_recording_context(context)
            self._recording_publication_in_progress = False
        refresh = getattr(self, "update_player_btn_is_paused", None)
        if callable(refresh):
            refresh()
        drain = getattr(self, "_drain_queued_directional_trigger", None)
        if callable(drain):
            drain()

    def _notify_process_recording_finished(self, session, *, context=None):
        context = context or self._recording_context_for_session(session)
        if context is None or context.cancelled or context.cleanup_owned:
            return
        completion = context.tcp_completion
        if completion is None or completion[0] != session.request.request_id:
            return
        # Claim before network I/O, including a send that raises after delivery.
        context.tcp_completion = None
        if completion[1] is not None:
            self._send_recording_tcp_finish(completion[1])

    def _on_process_recording_failed(self, session, failure):
        context = self._recording_context_for_session(session)
        if context is None:
            return
        self.default_logger.error(
            f"Recording failed request={session.request.request_id} "
            f"stage={failure.stage} path={session.request.path}: {failure.message}")
        context.failed = True
        context.validated_audio = None
        active = self._is_active_recording_process(session)
        if active:
            clear_source = getattr(self, "_clear_unpublished_recording_source", None)
            if callable(clear_source):
                clear_source()
            if getattr(session.request, "device", {}).get("backend") == VE_BACKEND:
                self._recording_ve_device = None
                self._recording_wav_calibration_metadata = None
        if context.cleanup_owned:
            # Abort/close already chose the controls, history and stage. A
            # cancellation that fails while closing still belongs to that
            # caller; diagnose it without applying ordinary failure recovery.
            return
        context.final = True
        context.accepted_audio = None
        self._drop_recording_context(context)
        if active:
            self.streaming_processor = None
            self._handle_invalid_recording(f"录音失败 ({failure.stage}): {failure.message}")

    def _on_process_recording_cancelled(self, session, descriptor):
        context = self._recording_context_for_session(session)
        if context is None or context.cleanup_owned:
            return
        context.final = True
        context.cancelled = True
        context.validated_audio = None
        active = self._is_active_recording_process(session)
        if active:
            clear_source = getattr(self, "_clear_unpublished_recording_source", None)
            if callable(clear_source):
                clear_source()
        context.accepted_audio = None
        self._drop_recording_context(context)
        if not active:
            return
        self.streaming_processor = None
        handled = self._cleanup_failed_recording_initialization("录音已取消")
        if handled:
            return
        discard = getattr(self, "_discard_current_recent_session", None)
        if callable(discard):
            discard()

    def _cancel_process_recording(self):
        session = getattr(self, "_recording_process_session", None)
        if session is None:
            return
        context = self._recording_context_for_session(session)
        if context is None:
            return
        context.preview_enabled = False
        context.final = True
        context.accepted_audio = None
        context.cancelled = True
        context.validated_audio = None
        context.cleanup_owned = True
        self._recording_process_preview_enabled = False
        self._recording_process_final = True
        self._recording_process_audio = None
        self._recording_process_cancelled = True
        self._recording_process_validated_audio = None
        clear_source = getattr(self, "_clear_unpublished_recording_source", None)
        if callable(clear_source):
            clear_source()
        if getattr(session.request, "device", {}).get("backend") == VE_BACKEND:
            self._recording_ve_device = None
            self._recording_wav_calibration_metadata = None
        # Resource cleanup callers (serial abort, close, replacement) decide
        # their own controls/history disposition. A late cancellation cannot
        # run ordinary completion cleanup over that decision.
        session.cancel()

    def _clear_unpublished_recording_source(self):
        # An abort must not detach provenance from retained previous audio.
        data = getattr(self, "data_struct", None)
        if (data is not None and getattr(data, "store_wave_data", None) is None
                and getattr(data, "store_wave_data_multi", None) is None):
            clear = getattr(self, "_clear_imported_wav_calibration_state", None)
            if callable(clear):
                clear()
            else:
                from base.data_struct.data_deal_struct import DataDealStruct
                DataDealStruct.clear_wav_calibration_context(data)

    def _recording_path_is_leased(self, path):
        bridge = getattr(self, "recording_bridge", None)
        return bridge is not None and bridge.service.is_path_leased(path)
