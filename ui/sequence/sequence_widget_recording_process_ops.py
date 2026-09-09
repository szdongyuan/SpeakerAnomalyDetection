"""Main-recording adapter: request snapshots, envelopes and accepted results."""
import copy
import os
import time
from uuid import uuid4

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from base.play_and_record import resolve_startup_trim_samples
from base.recording_preview_config import resolve_recording_preview_time_mode
from base.recording_process_protocol import FrozenConfig, RecordingFailure, RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from base.recording_settings import merge_audio_validation_thresholds
from base.ve3668n_input import validate_sample_rate
from base.ve3668n_wav_metadata import validate_ve_wav_metadata
from consts.ve3668n_consts import VE_BACKEND
from ui.recording_service_bridge import RecordingProcessorFacade, RecordingServiceBridge
from ui.sequence.recording_process_context import RecordingProcessContext
from ui.sequence.recording_analysis_eligibility import (
    enabled_analysis_identifiers,
    fast_recording_overlap_eligible,
)


class _RequestScopedDispatchFailureBridge(QObject):
    """Move a worker dispatch failure onto the owning GUI thread."""

    failed = pyqtSignal(str)

    def __init__(self, owner, generation):
        super().__init__(owner if isinstance(owner, QObject) else None)
        self._owner = owner
        self._generation = int(generation)
        self.failed.connect(self._deliver)

    def notify(self, request_id):
        self.failed.emit(str(request_id))

    def _deliver(self, request_id):
        owner = self._owner
        executor = getattr(owner, "_request_scoped_recording_executor", None)
        owner._reconcile_request_scoped_dispatch_failure(
            request_id, executor, self._generation)


class SequenceWidgetRecordingProcessOpsMixin:
    def _request_scoped_recording_generation(self):
        generation = getattr(self, "_request_scoped_executor_generation", None)
        if generation is None:
            generation = 0
            self._request_scoped_executor_generation = generation
        return int(generation)

    def _request_scoped_generation_is_current(self, generation):
        return (
            not bool(getattr(self, "_request_scoped_executor_closed", False))
            and int(generation) == self._request_scoped_recording_generation()
            and not any(bool(getattr(self, name, False)) for name in (
                "_closing", "_shutdown_started", "_recording_closed")))

    def _get_request_scoped_recording_executor(self):
        if (bool(getattr(self, "_request_scoped_executor_closed", False))
                or any(bool(getattr(self, name, False)) for name in (
                    "_closing", "_shutdown_started", "_recording_closed"))):
            return None
        executor = getattr(self, "_request_scoped_recording_executor", None)
        if executor is not None:
            return executor
        bridge = getattr(self, "recording_bridge", None)
        invoke = getattr(bridge, "_invoke", None)
        emit = getattr(invoke, "emit", None)
        if not callable(emit):
            return None
        from ui.sequence.request_scoped_recording_executor import (
            RequestScopedRecordingExecutor,
        )
        if getattr(self, "_request_scoped_durable_effect_lock", None) is None:
            import threading
            self._request_scoped_durable_effect_lock = threading.RLock()
        generation = self._request_scoped_recording_generation()
        notifier = _RequestScopedDispatchFailureBridge(self, generation)
        executor = RequestScopedRecordingExecutor(
            dispatch=emit, capacity=2, max_workers=1,
            dispatch_failure_notify=notifier.notify)
        self._request_scoped_dispatch_failure_bridge = notifier
        self._request_scoped_recording_executor = executor
        return executor

    def _reserve_request_scoped_recording_capacity(self, request_id):
        executor = self._get_request_scoped_recording_executor()
        if executor is None:
            # Isolated legacy hosts without a GUI dispatcher cannot run the
            # background publisher. Preserve their synchronous test boundary;
            # production RecordingServiceBridge instances always provide it.
            return None, "accepted"
        reserve = getattr(executor, "reserve_with_status", None)
        if not callable(reserve):
            return None, "accepted"
        return executor, str(reserve(request_id))

    def _release_recording_publication_reservation(self, context):
        if context is None or not bool(getattr(
                context, "publication_reservation_active", False)):
            return False
        # Claim before crossing into the executor so duplicate/late terminal
        # callbacks cannot release a newer request with the same owner state.
        context.publication_reservation_active = False
        executor = getattr(context, "publication_reservation_executor", None)
        release = getattr(executor, "release_reservation", None)
        return bool(release(context.request.request_id)) if callable(release) else False

    def _reconcile_request_scoped_dispatch_failure(
            self, request_id, executor, generation):
        if (executor is None
                or not self._request_scoped_generation_is_current(generation)
                or executor is not getattr(
                    self, "_request_scoped_recording_executor", executor)):
            return ()
        context = self._recording_contexts().get(str(request_id))
        if (context is None or context.publication_delivered
                or context.publication_executor_generation != int(generation)):
            return ()
        reconcile = getattr(executor, "reconcile_dispatch_failures", None)
        if not callable(reconcile):
            return ()
        return reconcile(str(request_id))

    def _deliver_request_scoped_recording_publication(
            self, context, outcome, generation=None):
        if generation is None:
            generation = context.publication_executor_generation
        if not self._request_scoped_generation_is_current(generation):
            return
        if context.publication_delivered or context.publication_delivery_finalizing:
            return
        context.publication_delivery_finalizing = True
        context.publication_started = False
        succeeded = outcome.error is None and outcome.value is True
        if not succeeded:
            if outcome.error is not None:
                context.business_completed = False
                context.business_failure = str(outcome.error)
                self.default_logger.error(
                    "Request-scoped recording publication failed "
                    f"request={context.request.request_id}: {outcome.error}")
            now = time.monotonic()
            from ui.sequence.request_scoped_recording_executor import (
                RequestScopedDispatchError,
            )
            can_retry = (
                not isinstance(outcome.error, RequestScopedDispatchError)
                and not context.cancelled and not context.failed
                and context.publication_attempts < context.publication_max_attempts
                and now <= context.publication_retry_deadline
                and not any(bool(getattr(self, name, False)) for name in (
                    "_closing", "_shutdown_started", "_recording_closed")))
            if can_retry and not context.publication_retry_pending:
                context.publication_retry_pending = True
                delay_ms = 50 * (2 ** max(0, context.publication_attempts - 1))
                scheduler = getattr(
                    self, "_schedule_request_scoped_recording_retry", None)
                if callable(scheduler):
                    scheduler(delay_ms, lambda context=context, generation=generation:
                              self._retry_request_scoped_recording_publication(
                                  context, generation))
                else:
                    QTimer.singleShot(
                        delay_ms,
                        lambda context=context, generation=generation:
                        self._retry_request_scoped_recording_publication(
                            context, generation))
                context.publication_delivery_finalizing = False
                return
        owned_active = self._recording_context_owns_active_workflow(context)
        stage = "upsert"
        try:
            session = context.session
            session_record = context.pending_ui_session_record
            if isinstance(session_record, dict):
                records = getattr(self, "recent_test_session_by_id", None)
                session_id = str(
                    context.recent_session_id
                    or session_record.get("session_id") or "")
                if isinstance(records, dict) and session_id:
                    existing = records.get(session_id)
                    if isinstance(existing, dict):
                        existing.clear()
                        existing.update(copy.deepcopy(session_record))
                    else:
                        records[session_id] = copy.deepcopy(session_record)
                panel = getattr(self, "recent_session_panel", None)
                upsert = getattr(panel, "upsert_session", None)
                if callable(upsert):
                    upsert(session_record)
            context.pending_ui_session_record = None
            stage = "present"
            if succeeded:
                presenter = getattr(
                    self, "_present_request_scoped_recording_context", None)
                if callable(presenter) and owned_active:
                    presenter(context)
            else:
                self._apply_active_publication_failure(
                    context,
                    context.business_failure
                    or "录音结果发布失败，已恢复录音控制")
            stage = "notify"
            if succeeded and session is not None and not context.cancelled:
                self._notify_process_recording_finished(session, context=context)
            stage = "drop"
            self._drop_recording_context(context)
        except Exception as error:
            self._recover_request_scoped_delivery_failure(
                context, stage, error, owned_active=owned_active)
            return
        context.publication_audio = None
        context.publication_final_windows = None
        context.publication_delivered = True
        context.publication_delivery_finalizing = False

    def _recover_request_scoped_delivery_failure(
            self, context, stage, error, *, owned_active):
        """Finish one failed GUI terminal without leaving admission wedged."""
        message = f"{stage} delivery failed: {error}"
        context.business_completed = False
        context.business_failure = message
        self.default_logger.error(
            "Request-scoped recording owner delivery failed "
            f"request={context.request.request_id} stage={stage}: {error}")
        try:
            self._apply_active_publication_failure(
                context, message, owned_active=owned_active)
        except Exception as recovery_error:
            self.default_logger.error(
                "Request-scoped active failure recovery callback failed "
                f"request={context.request.request_id}: {recovery_error}")
            if owned_active:
                context.active_transition_applied = True
                self.streaming_processor = None
                self.player_status_flag = False
        dropped = False
        try:
            dropped = self._drop_recording_context(context) is not False
        except Exception as drop_error:
            self.default_logger.error(
                "Request-scoped context drop recovery failed "
                f"request={context.request.request_id}: {drop_error}")
            contexts = self._recording_contexts()
            if contexts.get(context.request.request_id) is context:
                contexts.pop(context.request.request_id, None)
                if getattr(self, "_active_recording_process_id", None) == (
                        context.request.request_id):
                    self._active_recording_process_id = None
                self._record_workflow_busy = bool(contexts)
                dropped = True
        context.pending_ui_session_record = None
        context.publication_audio = None
        context.publication_final_windows = None
        context.publication_started = False
        context.publication_delivered = True
        context.publication_delivery_finalizing = False
        release_reservation = getattr(
            self, "_release_recording_publication_reservation", None)
        if dropped and callable(release_reservation):
            release_reservation(context)

    def _retry_request_scoped_recording_publication(
            self, context, generation=None):
        if generation is None:
            generation = context.publication_executor_generation
        if not self._request_scoped_generation_is_current(generation):
            return
        context.publication_retry_pending = False
        if context.publication_delivered or context.cancelled:
            return
        if self._recording_contexts().get(context.request.request_id) is not context:
            return
        self._submit_request_scoped_recording_publication(context)

    def _retry_request_scoped_recording_submission(
            self, context, generation=None):
        if generation is None:
            generation = context.publication_executor_generation
        # A callback queued before close must be a complete no-op.  In
        # particular, do not clear flags or call the executor factory after
        # its host generation has been invalidated.
        if not self._request_scoped_generation_is_current(generation):
            return
        context.publication_submission_retry_pending = False
        if context.publication_delivered or context.cancelled or context.failed:
            return
        if self._recording_contexts().get(context.request.request_id) is not context:
            return
        if time.monotonic() > context.publication_retry_deadline:
            self._fail_request_scoped_recording_submission(
                context,
                "request-scoped recording result executor deadline expired")
            return
        waiters = getattr(self, "_request_scoped_submission_waiters", None)
        if isinstance(waiters, list) and waiters and waiters[0] is not context:
            self._schedule_request_scoped_submission_front(generation)
            return
        self._submit_request_scoped_recording_publication(context)

    def _remove_request_scoped_submission_waiter(self, context, generation=None):
        waiters = getattr(self, "_request_scoped_submission_waiters", None)
        if not isinstance(waiters, list):
            return
        self._request_scoped_submission_waiters = [
            value for value in waiters if value is not context]
        if generation is not None:
            self._schedule_request_scoped_submission_front(generation)

    def _schedule_request_scoped_submission_front(self, generation):
        waiters = getattr(self, "_request_scoped_submission_waiters", None)
        if not isinstance(waiters, list):
            return
        contexts = self._recording_contexts()
        waiters[:] = [
            context for context in waiters
            if (not context.publication_delivered and not context.cancelled
                and contexts.get(context.request.request_id) is context)]
        if not waiters or not self._request_scoped_generation_is_current(generation):
            return
        context = waiters[0]
        if context.publication_submission_retry_pending:
            return
        now = time.monotonic()
        if now > context.publication_retry_deadline:
            self._fail_request_scoped_recording_submission(
                context,
                "request-scoped recording result executor deadline expired")
            return
        context.publication_submission_retry_pending = True
        delay_ms = min(250, 50 * (2 ** min(
            4, max(0, context.publication_submission_failures - 1))))
        scheduler = getattr(
            self, "_schedule_request_scoped_recording_retry", None)
        callback = lambda context=context, generation=generation: (
            self._retry_request_scoped_recording_submission(context, generation))
        if callable(scheduler):
            scheduler(delay_ms, callback)
        else:
            QTimer.singleShot(delay_ms, callback)

    def _apply_active_publication_failure(
            self, context, message, *, owned_active=None):
        if owned_active is None:
            owned_active = self._recording_context_owns_active_workflow(context)
        if (context.active_transition_applied
                or not owned_active):
            return False
        context.active_transition_applied = True
        self.streaming_processor = None
        self.player_status_flag = False
        handler = getattr(self, "_handle_invalid_recording", None)
        if callable(handler):
            handler(str(message))
        return True

    def _fail_request_scoped_recording_submission(self, context, message):
        if context.publication_delivered:
            return
        context.publication_started = False
        context.publication_submission_retry_pending = False
        context.publication_delivered = True
        context.business_completed = False
        context.business_failure = str(message)
        self.default_logger.error(
            "Request-scoped recording result publication failed "
            f"request={context.request.request_id}: {message}")
        generation = self._request_scoped_recording_generation()
        self._remove_request_scoped_submission_waiter(
            context,
            generation if self._request_scoped_generation_is_current(
                generation) else None)
        self._apply_active_publication_failure(context, message)
        self._drop_recording_context(context)
        context.publication_audio = None
        context.publication_final_windows = None

    def _handle_request_scoped_submission_full(self, context, generation):
        context.publication_submission_failures += 1
        if (context.cancelled or context.failed
                or not self._request_scoped_generation_is_current(generation)):
            return
        waiters = getattr(self, "_request_scoped_submission_waiters", None)
        if not isinstance(waiters, list):
            waiters = []
            self._request_scoped_submission_waiters = waiters
        if not any(value is context for value in waiters):
            waiters.append(context)
        self._schedule_request_scoped_submission_front(generation)

    def _submit_request_scoped_recording_publication(self, context):
        if context.publication_started or context.publication_delivered:
            return False
        generation = self._request_scoped_recording_generation()
        if not self._request_scoped_generation_is_current(generation):
            return False
        waiters = getattr(self, "_request_scoped_submission_waiters", None)
        if isinstance(waiters, list) and waiters:
            if not any(value is context for value in waiters):
                waiters.append(context)
            if (waiters[0] is not context
                    or context.publication_submission_retry_pending):
                self._schedule_request_scoped_submission_front(generation)
                return False
        executor = (getattr(context, "publication_reservation_executor", None)
                    or self._get_request_scoped_recording_executor())
        if executor is None or context.publication_audio is None:
            return False
        audio = context.publication_audio
        kwargs = dict(
            recorded_mono=audio.mono, recorded_multi=audio.multi,
            sample_rate=context.publication_sample_rate,
            completion_source="process", prefinalized=True,
            final_waveform_windows=context.publication_final_windows,
            recording_context=context, request_owned_only=True)
        context.publication_started = True
        context.publication_executor_generation = generation
        submit_with_status = getattr(executor, "submit_with_status", None)
        submit = submit_with_status if callable(submit_with_status) else executor.submit
        submit_result = submit(
            context.request.request_id,
            lambda kwargs=kwargs, generation=generation:
            self._on_streaming_complete(**kwargs)
            if self._request_scoped_generation_is_current(generation) else False,
            lambda outcome, context=context, generation=generation:
            self._deliver_request_scoped_recording_publication(
                context, outcome, generation))
        if callable(submit_with_status):
            submit_status = str(submit_result)
        else:
            submit_status = "accepted" if submit_result else "full"
        context.publication_submit_status = submit_status
        if submit_status != "accepted":
            context.publication_started = False
            if submit_status == "full":
                self._handle_request_scoped_submission_full(context, generation)
            return False
        remove_waiter = getattr(
            self, "_remove_request_scoped_submission_waiter", None)
        if callable(remove_waiter):
            remove_waiter(context, generation)
        context.publication_attempts += 1
        context.publication_submission_retry_pending = False
        return True

    def _publish_serialized_recording_context(self, context):
        """Finish an ineligible request through the pre-overlap UI workflow.

        Admission guarantees that a context containing any analysis outside the
        fast-overlap set cannot coexist with another result session.  Keeping
        that request on the synchronous completion path preserves the original
        analysis widgets and user-visible behavior without teaching the
        request-scoped background analyzer about additional analysis types.
        """
        if context.publication_started or context.publication_delivered:
            return False
        audio = context.publication_audio
        if audio is None:
            return False
        context.publication_started = True
        context.publication_attempts += 1
        try:
            succeeded = self._on_streaming_complete(
                recorded_mono=audio.mono,
                recorded_multi=audio.multi,
                sample_rate=context.publication_sample_rate,
                completion_source="process",
                prefinalized=True,
                final_waveform_windows=context.publication_final_windows,
                recording_context=context,
                serialized_legacy=True,
            )
            context.business_completed = succeeded is True
            if succeeded is True and not context.cancelled:
                self._notify_process_recording_finished(
                    context.session, context=context)
        finally:
            context.publication_started = False
            context.publication_delivered = True
            context.active_transition_applied = True
            finalize_channels = getattr(
                self, "_finalize_recording_channel_selection", None)
            if callable(finalize_channels):
                finalize_channels()
            self._drop_recording_context(context)
            context.publication_audio = None
            context.publication_final_windows = None
        return True

    def _shutdown_request_scoped_recording_executor(self):
        self._request_scoped_executor_generation = (
            self._request_scoped_recording_generation() + 1)
        self._request_scoped_executor_closed = True
        self._request_scoped_submission_waiters = []
        executor = getattr(self, "_request_scoped_recording_executor", None)
        self._request_scoped_recording_executor = None
        self._request_scoped_dispatch_failure_bridge = None
        retained = ()
        if executor is not None:
            retained = executor.shutdown(wait=True, timeout=.5)
            for context in self._recording_contexts().values():
                if getattr(context, "publication_reservation_executor", None) is executor:
                    context.publication_reservation_active = False
            if retained:
                self.default_logger.warning(
                    "Request-scoped result shutdown retained work: "
                    + ", ".join(retained))
        durable_lock = getattr(self, "_request_scoped_durable_effect_lock", None)
        if durable_lock is not None:
            acquired = durable_lock.acquire(timeout=.5)
            if acquired:
                durable_lock.release()
            else:
                self._request_scoped_skip_final_spool = True
                retained = tuple(retained) + ("durable-effect-boundary",)
                self.default_logger.error(
                    "Request-scoped durable publication did not quiesce before close")
        self._request_scoped_retained_work_ids = tuple(retained)
        return tuple(retained)

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

    def _recording_admission_config_snapshot(self):
        builder = getattr(self, "_build_recent_session_config_snapshot", None)
        snapshot = copy.deepcopy(builder() or {}) if callable(builder) else {}
        if not isinstance(snapshot.get("analysis_config"), dict):
            snapshot["analysis_config"] = copy.deepcopy(
                getattr(self, "analysis_config", {}) or {})
        return snapshot

    def _unfinished_recording_analysis_identifiers(self):
        identifiers = []
        contexts = getattr(self, "_recording_process_contexts", None)
        for context in (contexts.values() if isinstance(contexts, dict) else ()):
            frozen = tuple(getattr(context, "enabled_analysis_identifiers", ()) or ())
            if not frozen:
                frozen = enabled_analysis_identifiers(
                    getattr(context, "recent_session_config_snapshot", {}) or {})
            identifiers.append(frozen)
        return tuple(identifiers)

    def _recording_analysis_overlap_eligible(self, config_snapshot=None):
        snapshot = (SequenceWidgetRecordingProcessOpsMixin
                    ._recording_admission_config_snapshot(self)
                    if config_snapshot is None else copy.deepcopy(config_snapshot))
        return fast_recording_overlap_eligible(
            enabled_analysis_identifiers(snapshot),
            SequenceWidgetRecordingProcessOpsMixin
            ._unfinished_recording_analysis_identifiers(self),
        )

    def _can_start_recording_workflow(self, config_snapshot=None):
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
        bridge = getattr(self, "recording_bridge", None)
        overlap_eligible = (SequenceWidgetRecordingProcessOpsMixin
                            ._recording_analysis_overlap_eligible(
                                self, config_snapshot))
        if bridge is not None:
            service = bridge.service
            if not overlap_eligible:
                return (not getattr(service, "busy", False)
                        and not getattr(self, "_record_workflow_busy", False)
                        and not getattr(self, "player_status_flag", False))
            try:
                service_can_start = service.can_start_recording
            except AttributeError:
                # Compatibility is intentionally limited to legacy test/window
                # service doubles that predate the additive property.
                return (not getattr(service, "busy", False)
                        and not getattr(self, "_record_workflow_busy", False)
                        and not getattr(self, "player_status_flag", False))
            if not service_can_start:
                return False
            executor = getattr(self, "_request_scoped_recording_executor", None)
            if (executor is not None
                    and not bool(getattr(executor, "can_reserve", True))):
                return False
        contexts = getattr(self, "_recording_process_contexts", None)
        contexts = contexts if isinstance(contexts, dict) else {}
        if getattr(self, "_record_workflow_busy", False) and not contexts:
            return False
        processor = getattr(self, "streaming_processor", None)
        owned_processors = {id(context.processor) for context in contexts.values()
                            if context.processor is not None}
        if getattr(self, "player_status_flag", False):
            if processor is None or id(processor) not in owned_processors:
                return False
        return True

    def _can_start_calibration_workflow(self):
        """Evaluate an empty-analysis calibration against unfinished contexts."""
        calibration_snapshot = FrozenConfig.snapshot({
            "analysis_config": {"display_sequence": []},
        })
        return self._can_start_recording_workflow(
            config_snapshot=calibration_snapshot)

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
        release_reservation = getattr(
            self, "_release_recording_publication_reservation", None)
        if callable(release_reservation):
            release_reservation(context)
        elif bool(getattr(context, "publication_reservation_active", False)):
            # Narrow compatibility for isolated callback harnesses that bind
            # only `_drop_recording_context` from this mixin.
            context.publication_reservation_active = False
            executor = getattr(context, "publication_reservation_executor", None)
            release = getattr(executor, "release_reservation", None)
            if callable(release):
                release(request_id)
        contexts.pop(request_id, None)
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
        bridge.analysis_eligibility_provider = (
            self._unfinished_recording_analysis_identifiers)
        return bridge

    def _start_process_recording(self, recorded_dict, sample_rate, *, tcp_completion_address=None):
        config_snapshot = getattr(self, "_pending_recording_config_snapshot", None)
        self._pending_recording_config_snapshot = None
        if config_snapshot is None:
            config_snapshot = self._recording_admission_config_snapshot()
        contexts = self._recording_contexts()
        eligible = self._recording_analysis_overlap_eligible(config_snapshot)
        bridge = getattr(self, "recording_bridge", None)
        if contexts and not eligible:
            raise RuntimeError(
                "recording result processing blocks this analysis configuration")
        if (bridge is not None and not eligible
                and getattr(bridge.service, "busy", False)):
            # Unlike capacity, analysis eligibility is UI-owned policy and is
            # not represented by RecordingService.start(). A reentrant result
            # transition after the initial guard must therefore reject
            # explicitly so the caller performs its established full cleanup.
            raise RuntimeError(
                "recording service is busy with ineligible result processing")
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
        reservation_executor, reservation_status = (
            self._reserve_request_scoped_recording_capacity(request.request_id))
        if reservation_status != "accepted":
            if reservation_status == "full":
                raise RuntimeError(
                    "CAPACITY_BACKPRESSURE: recording publication pipeline is full")
            raise RuntimeError(
                "recording publication pipeline rejected reservation: "
                f"{reservation_status}")
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
                product_condition_key=str(
                    getattr(self, "_get_active_product_condition_key", lambda: "")() or ""),
                product_group_id=str(
                    getattr(self, "_manual_product_condition_group_id", "") or ""),
                publication_group_id=str(
                    getattr(self, "_manual_product_condition_group_id", "")
                    or getattr(self, "_current_cycle_recorded_count", "")
                    or getattr(self, "_current_run_recording_token", "")
                    or request.request_id),
                product_condition_keys=tuple(
                    getattr(self, "_manual_product_condition_keys", lambda: ())() or ()),
                manual_product_cycle_active=bool(
                    getattr(self, "_is_manual_product_condition_cycle_active", lambda: False)()),
                serial_product_condition_executing=bool(
                    getattr(self, "_serial_product_condition_executing", False)),
                directional_cycle_active=bool(
                    getattr(self, "_is_directional_cycle_active", lambda: False)()),
                count_mode=str(
                    getattr(getattr(self, "count_board", None), "mode", "") or ""),
                barcode=str(
                    (getattr(self, "recorded_signal_info", {}) or {}).get("barcode") or ""),
                analysis_result_dict={},
                recent_session_config_snapshot=copy.deepcopy(config_snapshot),
                enabled_analysis_identifiers=enabled_analysis_identifiers(config_snapshot),
                product_report_config=dict(
                    getattr(self, "product_test_pdf_report_config", {}) or {}),
                analysis_required=bool(
                    getattr(self, "_should_run_silent_analysis_after_recording", lambda: False)()),
                stimulus_info=dict(
                    getattr(getattr(self, "data_struct", None), "stimulus_info", {}) or {}),
                stimulus_signal=(
                    np.asarray(
                        getattr(getattr(self, "data_struct", None), "stimulus_data"),
                        dtype=np.float32,
                    ).copy()
                    if getattr(getattr(self, "data_struct", None), "stimulus_data", None)
                    is not None else None),
                analysis_executor=(
                    getattr(self, "_recording_request_analysis_executor", None)
                    if callable(getattr(
                        self, "_recording_request_analysis_executor", None))
                    else None),
                publication_reservation_executor=reservation_executor,
                publication_reservation_active=reservation_executor is not None,
                publication_executor_generation=(
                    self._request_scoped_recording_generation()
                    if reservation_executor is not None else -1),
            )
            validate_workspace = getattr(
                self, "_validate_final_waveform_workspace", None)
            if callable(validate_workspace):
                try:
                    context.final_windows = validate_workspace(request.channels)
                except (TypeError, ValueError, OverflowError):
                    context.final_windows = None
        except Exception:
            if reservation_executor is not None:
                reservation_executor.release_reservation(request.request_id)
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
            if getattr(session.request, "device", {}).get("backend") == VE_BACKEND:
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
            if (getattr(session.request, "device", {}).get("backend") == VE_BACKEND
                    and audio is not context.validated_audio):
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

    def _freeze_request_scoped_publication_inputs(self, context):
        """Copy UI-owned publication inputs before background execution."""
        if context.publication_owner_snapshot_frozen:
            return
        records = getattr(self, "recent_test_session_by_id", None)
        context.publication_recent_sessions_snapshot = copy.deepcopy(
            records if isinstance(records, dict) else {})
        context.recent_session_config_snapshot = copy.deepcopy(
            context.recent_session_config_snapshot or {})
        context.product_report_config = copy.deepcopy(
            context.product_report_config or {})
        context.condition_record_cache = copy.deepcopy(
            context.condition_record_cache or {})
        context.publication_owner_snapshot_frozen = True

    def _publish_recording_context(self, context):
        session = context.session
        if session is None or context.cancelled or context.failed:
            return
        audio = context.accepted_audio
        if getattr(session.request, "device", {}).get("backend") == VE_BACKEND:
            if session.request is not context.request:
                self._on_process_recording_failed(session, RecordingFailure(
                    session.request.request_id, "protocol", session.request.path,
                    "VE accepted result no longer matches its frozen request"))
                return
            if (not session.released.is_set() or session.release_error is not None
                    or audio is None or audio is not context.validated_audio):
                return
        if audio is not None and session.state == "completed":
            context.accepted_audio = None
            context.validated_audio = None
            for warning in audio.descriptor.warnings:
                self.default_logger.warning(f"Recording {session.request.request_id}: {warning}")
            context.publication_audio = audio
            context.publication_sample_rate = int(audio.descriptor.sample_rate)
            context.publication_final_windows = context.final_windows
            freeze_inputs = getattr(
                self, "_freeze_request_scoped_publication_inputs", None)
            if callable(freeze_inputs):
                freeze_inputs(context)
            else:
                # Narrow compatibility for isolated callback harnesses that
                # bind only this method rather than the complete mixin.
                context.publication_owner_snapshot_frozen = True
            if (not fast_recording_overlap_eligible(
                    context.enabled_analysis_identifiers, ())):
                self._publish_serialized_recording_context(context)
                return
            if not context.publication_retry_deadline:
                context.publication_retry_deadline = time.monotonic() + 5.0
            if not self._submit_request_scoped_recording_publication(context):
                if context.publication_submit_status == "full":
                    self.default_logger.warning(
                        "Request-scoped recording result executor is at capacity; "
                        f"resubmission retained request={context.request.request_id}")

    def _notify_process_recording_finished(self, session, *, context=None):
        context = context or self._recording_context_for_session(session)
        if context is None or context.cancelled or context.cleanup_owned:
            return
        completion = context.tcp_completion
        if completion is None or completion[0] != session.request.request_id:
            return
        # Claim before network I/O, including a send that raises after delivery.
        context.tcp_completion = None
        context.business_effect_attempts["tcp:authorization"] = (
            context.business_effect_attempts.get("tcp:authorization", 0) + 1)
        context.business_effect_ledger["tcp:authorization"] = completion[1]
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
