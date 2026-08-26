"""Recording-owned imported-audio load and cancellation transaction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from PyQt5 import sip
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from ui.sequence.sequence_event_bus import ImportTerminalRecipientResult
from ui.sequence.sequence_messages import (
    CancelImportedAudioRequested,
    ImportedAudioFailed,
    ImportedAudioReady,
    LoadImportedAudioRequested,
    ResourceLifecycleRequested,
)
from ui.sequence.sequence_recording_import_service import (
    AudioImportFailure,
    ImportedAudioStage,
    SequenceImportedAudioService,
    mutable_import_snapshot,
)
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_view import SequenceRecordingImportView


@dataclass(frozen=True, slots=True)
class ImportTransactionRecovery:
    runtime_restored: bool
    projection_restored: bool
    converged_to_empty: bool = False
    consistent: bool = True
    failures: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.runtime_restored and self.projection_restored


@dataclass(frozen=True, slots=True)
class _PendingImportTerminal:
    import_id: str
    signal: Any
    event: Any


@dataclass(slots=True, eq=False)
class _ImportTransactionToken:
    import_id: str
    lifecycle_generation: int
    model_begun: bool = False
    mutation_started: bool = False
    terminal_started: bool = False
    terminal_acknowledged: bool = False
    invalidated: bool = False


@dataclass(slots=True, eq=False)
class _ImportTerminalAttempt:
    delivery_id: tuple[str, str]


class _ImportLifecycleClosed(RuntimeError):
    """Internal control flow for a reentrant disconnect or native deletion."""


class _ImportTerminalLifecycle:
    """Abandon one pending delivery without accessing a destroyed controller."""

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._lock = RLock()
        self._delivery_id: tuple[str, str] | None = None
        self._inflight_attempt: object | None = None
        self._closed = False

    def set_pending(self, delivery_id: tuple[str, str]) -> None:
        with self._lock:
            if not self._closed:
                self._delivery_id = delivery_id

    def begin_delivery(
        self, delivery_id: tuple[str, str], attempt: object
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._delivery_id = delivery_id
            self._inflight_attempt = attempt

    def end_delivery(
        self,
        delivery_id: tuple[str, str],
        attempt: object,
        *,
        acknowledged: bool,
    ) -> None:
        abandon_id = None
        with self._lock:
            if self._inflight_attempt is not attempt:
                return
            self._inflight_attempt = None
            if acknowledged:
                if self._delivery_id == delivery_id:
                    self._delivery_id = None
                return
            if self._closed and self._delivery_id == delivery_id:
                abandon_id = self._delivery_id
                self._delivery_id = None
        if abandon_id is not None:
            self._abandon(abandon_id)

    def complete(self) -> None:
        with self._lock:
            self._delivery_id = None

    def _abandon(self, delivery_id: tuple[str, str]) -> None:
        abandon = getattr(self._bus, "abandon_import_terminal", None)
        if not callable(abandon):
            return
        try:
            abandon(delivery_id, "recording-disconnect")
        except BaseException:
            # QObject destruction cannot propagate into the Qt event loop.
            return

    def close(self, *_args: Any) -> None:
        delivery_id = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._inflight_attempt is not None:
                return
            delivery_id = self._delivery_id
            self._delivery_id = None
        if delivery_id is not None:
            self._abandon(delivery_id)


class _WorkflowIdentityProviderFailure(RuntimeError):
    """Contain a failed coordinator identity read at the Qt slot boundary."""


class SequenceRecordingImportController(QObject):
    """Own one Workflow-admitted imported-audio load through terminal ack."""

    _import_terminal_retry_driver_requested = pyqtSignal()
    _import_terminal_timer_stop_requested = pyqtSignal()

    def __init__(
        self,
        model: RecordingModel,
        view: SequenceRecordingImportView | Any,
        *,
        bus: Any,
        runtime: Any = None,
        workflow_identity_provider: Callable[[], Mapping[str, Any]] | None = None,
        import_service: SequenceImportedAudioService | Any | None = None,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.view = view
        self.bus = bus
        self.runtime = runtime
        self.workflow_identity_provider = workflow_identity_provider
        self.import_service = import_service or SequenceImportedAudioService(
            logger=logger, reference_logger=logger
        )
        self.logger = logger
        self._state_lock = RLock()
        self._lifecycle_generation = 0
        self._active_import_transaction: _ImportTransactionToken | None = None
        self._active_import_terminal_attempt: _ImportTerminalAttempt | None = None
        self._pending_import_terminal: _PendingImportTerminal | None = None
        self._import_terminal_dispatch_active = True
        self._import_terminal_retry_attempt = 0
        self._import_terminal_retry_delay_ms = 0
        self._import_terminal_retry_base_delay_ms = 10
        self._import_terminal_retry_max_delay_ms = 1_000
        self._import_terminal_retry_driver_queued = False
        self._import_terminal_timer_stop_queued = False
        self._import_terminal_retry_timer = self._new_import_terminal_retry_timer()
        self._import_terminal_retry_driver_requested.connect(
            self._handle_import_terminal_retry_driver_requested,
            Qt.QueuedConnection,
        )
        self._import_terminal_timer_stop_requested.connect(
            self._handle_import_terminal_timer_stop_requested,
            Qt.QueuedConnection,
        )
        self._import_terminal_lifecycle = _ImportTerminalLifecycle(bus)
        self.destroyed.connect(self._import_terminal_lifecycle.close)

    @staticmethod
    def _native_alive(value: Any) -> bool:
        try:
            return value is not None and not sip.isdeleted(value)
        except (RuntimeError, TypeError):
            return False

    def _controller_native_alive(self) -> bool:
        return self._native_alive(self)

    def _controller_open(self) -> bool:
        if not self._controller_native_alive():
            return False
        with self._state_lock:
            return self._import_terminal_dispatch_active

    def _reserve_import_transaction(
        self, command: LoadImportedAudioRequested
    ) -> _ImportTransactionToken | None:
        if not self._controller_native_alive():
            return None
        with self._state_lock:
            if (
                not self._import_terminal_dispatch_active
                or self._active_import_transaction is not None
            ):
                return None
            token = _ImportTransactionToken(
                command.import_id, self._lifecycle_generation
            )
            self._active_import_transaction = token
            return token

    def _transaction_open(self, token: _ImportTransactionToken) -> bool:
        if not self._controller_native_alive():
            token.invalidated = True
            return False
        with self._state_lock:
            open_now = bool(
                self._import_terminal_dispatch_active
                and not token.invalidated
                and token.lifecycle_generation == self._lifecycle_generation
                and self._active_import_transaction is token
            )
        if not open_now:
            token.invalidated = True
        return open_now

    def _require_transaction_open(
        self, token: _ImportTransactionToken
    ) -> None:
        if not self._transaction_open(token):
            raise _ImportLifecycleClosed("import lifecycle closed")

    def _release_import_transaction(
        self, token: _ImportTransactionToken
    ) -> None:
        with self._state_lock:
            if self._active_import_transaction is token:
                self._active_import_transaction = None

    def _begin_import_terminal_attempt(
        self, delivery_id: tuple[str, str]
    ) -> _ImportTerminalAttempt | None:
        with self._state_lock:
            if self._active_import_terminal_attempt is not None:
                return None
            attempt = _ImportTerminalAttempt(delivery_id)
            self._active_import_terminal_attempt = attempt
            return attempt

    def _end_import_terminal_attempt(
        self, attempt: _ImportTerminalAttempt
    ) -> None:
        with self._state_lock:
            if self._active_import_terminal_attempt is attempt:
                self._active_import_terminal_attempt = None

    def _import_terminal_attempt_is_active(self) -> bool:
        with self._state_lock:
            return self._active_import_terminal_attempt is not None

    def _new_import_terminal_retry_timer(self) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self.handle_import_terminal_retry_timeout)
        return timer

    def _owner_thread_is_current(self) -> bool:
        if not self._controller_native_alive():
            return False
        try:
            return QThread.currentThread() == self.thread()
        except (RuntimeError, TypeError):
            return False

    def _request_import_terminal_retry_driver(self) -> bool:
        if not self._controller_open():
            return False
        with self._state_lock:
            if self._import_terminal_retry_driver_queued:
                return True
            self._import_terminal_retry_driver_queued = True
        try:
            self._import_terminal_retry_driver_requested.emit()
        except (RuntimeError, TypeError):
            with self._state_lock:
                self._import_terminal_retry_driver_queued = False
            return False
        return True

    def _request_import_terminal_timer_stop(self) -> bool:
        if not self._controller_native_alive():
            return False
        with self._state_lock:
            if self._import_terminal_timer_stop_queued:
                return True
            self._import_terminal_timer_stop_queued = True
        try:
            self._import_terminal_timer_stop_requested.emit()
        except (RuntimeError, TypeError):
            with self._state_lock:
                self._import_terminal_timer_stop_queued = False
            return False
        return True

    def _ensure_import_terminal_retry_timer(self) -> QTimer | None:
        timer = self._import_terminal_retry_timer
        if self._native_alive(timer):
            return timer
        if not self._controller_open():
            return None
        if not self._owner_thread_is_current():
            self._request_import_terminal_retry_driver()
            return None
        try:
            timer = self._new_import_terminal_retry_timer()
        except (RuntimeError, TypeError):
            return None
        self._import_terminal_retry_timer = timer
        return timer

    @pyqtSlot()
    def _handle_import_terminal_retry_driver_requested(self) -> None:
        with self._state_lock:
            self._import_terminal_retry_driver_queued = False
        if not self._controller_open() or self._pending_import_terminal is None:
            return
        self._schedule_import_terminal_retry()

    @pyqtSlot()
    def _handle_import_terminal_timer_stop_requested(self) -> None:
        with self._state_lock:
            self._import_terminal_timer_stop_queued = False
        self._safe_timer_stop()

    def _safe_timer_stop(self) -> bool:
        if not self._owner_thread_is_current():
            return self._request_import_terminal_timer_stop()
        timer = self._import_terminal_retry_timer
        if not self._native_alive(timer):
            return False
        try:
            timer.stop()
        except (RuntimeError, TypeError):
            return False
        return True

    def _retire_aborted_import(self, token: _ImportTransactionToken) -> None:
        if (
            not token.model_begun
            or (token.terminal_started and not token.terminal_acknowledged)
        ):
            return
        try:
            self.model.retire_import(token.import_id)
        except BaseException:
            return

    def _abort_closed_transaction(
        self,
        token: _ImportTransactionToken,
        runtime_checkpoint: tuple[dict[str, Any], dict[str, Any]] | None,
        projection_checkpoint: Any,
    ) -> bool:
        token.invalidated = True
        if token.mutation_started and runtime_checkpoint is not None:
            self._recover_import_transaction(
                runtime_checkpoint,
                projection_checkpoint,
                # The owner is already closed.  Restore the plain runtime
                # checkpoint, but do not re-enter any view/Qt callback.
                allow_projection=False,
                fallback_to_empty=False,
            )
        self._retire_aborted_import(token)
        self._release_import_transaction(token)
        return False

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def _workflow_identity(self) -> Mapping[str, Any] | None:
        if self.workflow_identity_provider is None:
            return None
        try:
            value = self.workflow_identity_provider()
        except BaseException as error:
            detail = self._bounded_error_text(error)
            self._log("error", f"recording import identity provider failed: {detail}")
            raise _WorkflowIdentityProviderFailure(
                f"recording import identity provider failed: {detail}"
            ) from None
        if not isinstance(value, Mapping):
            raise _WorkflowIdentityProviderFailure(
                "recording import identity provider returned a non-mapping value"
            )
        return value

    @staticmethod
    def _safe_error_text(error: BaseException) -> str:
        try:
            detail = str(error)
        except BaseException:
            detail = "<unprintable error>"
        return detail or type(error).__name__

    @classmethod
    def _bounded_error_text(cls, error: BaseException) -> str:
        detail = cls._safe_error_text(error)
        if len(detail) > 512:
            return f"{detail[:509]}..."
        return detail

    @pyqtSlot(object)
    def handle_load_imported_audio_requested(
        self, command: LoadImportedAudioRequested
    ) -> bool:
        """Stage and transactionally commit one Workflow-admitted import."""
        if type(command) is not LoadImportedAudioRequested:
            return False
        if not self._owner_thread_is_current():
            return False
        if not self._controller_open():
            return False
        if self.model.is_retired_import(command.import_id):
            self._log("warning", f"ignored retired audio import {command.import_id}")
            return False
        if self.model.active_import_id is not None:
            self._log("warning", f"ignored reentrant audio import {command.import_id}")
            return False
        token = self._reserve_import_transaction(command)
        if token is None:
            return False

        staging_completed = False
        checkpoints_ready = False
        runtime_checkpoint = None
        projection_checkpoint = None
        transaction_recovery: ImportTransactionRecovery | None = None
        try:
            try:
                accepted = self._accept_import_identity(command)
            except _WorkflowIdentityProviderFailure as error:
                self._require_transaction_open(token)
                self.model.begin_import(
                    command.import_id, command.workflow_generation
                )
                token.model_begun = True
                self._require_transaction_open(token)
                return self._finish_import_failure(
                    command,
                    self._safe_error_text(error),
                    token=token,
                    clear_runtime=False,
                )
            self._require_transaction_open(token)
            if not accepted:
                self._release_import_transaction(token)
                return False

            self.model.begin_import(command.import_id, command.workflow_generation)
            token.model_begun = True
            self._require_transaction_open(token)
            if not self.model.import_runtime_consistent:
                return self._finish_import_failure(
                    command,
                    "audio import runtime is inconsistent: "
                    f"{self.model.import_consistency_failure}",
                    token=token,
                    clear_runtime=False,
                )

            selected_path = self.view.choose_import_audio_path(
                command.selected_path
            )
            self._require_transaction_open(token)
            if self.model.import_cancel_pending:
                return self._finish_import_failure(
                    command,
                    self.model.import_cancellation_reason,
                    token=token,
                    clear_runtime=False,
                )
            if not selected_path:
                return self._finish_import_failure(
                    command,
                    "audio import was cancelled",
                    token=token,
                    clear_runtime=False,
                )

            if isinstance(self.import_service, SequenceImportedAudioService):
                stage = self.import_service.load(
                    command,
                    selected_path,
                    boundary_check=lambda: self._transaction_open(token),
                )
            else:
                stage = self.import_service.load(command, selected_path)
            self._require_transaction_open(token)
            staging_completed = True
            if self.model.import_cancel_pending:
                return self._finish_import_failure(
                    command,
                    self.model.import_cancellation_reason,
                    token=token,
                    clear_runtime=False,
                )
            if not self._import_identity_is_current(command):
                self._require_transaction_open(token)
                return self._finish_import_failure(
                    command,
                    "audio import became stale before commit",
                    token=token,
                    clear_runtime=False,
                )
            self._require_transaction_open(token)

            event = self._ready_import_event(command, stage)
            self._require_transaction_open(token)
            runtime_checkpoint = self._capture_import_runtime()
            self._require_transaction_open(token)
            projection_checkpoint = self._capture_import_projection()
            self._require_transaction_open(token)
            checkpoints_ready = True

            token.mutation_started = True
            try:
                self._commit_import_stage(stage, token=token)
                self._require_transaction_open(token)
                self._project_import_stage(stage, token=token)
                self._require_transaction_open(token)
                if not self._import_identity_is_current(command):
                    self._require_transaction_open(token)
                    transaction_recovery = self._recover_import_transaction(
                        runtime_checkpoint,
                        projection_checkpoint,
                        token=token,
                    )
                    token.mutation_started = False
                    return self._finish_import_failure(
                        command,
                        "audio import was cancelled before completion",
                        token=token,
                        clear_runtime=False,
                    )
                self._require_transaction_open(token)
            except _ImportLifecycleClosed:
                raise
            except BaseException:
                if not self._transaction_open(token):
                    raise _ImportLifecycleClosed(
                        "import lifecycle closed during projection"
                    ) from None
                transaction_recovery = self._recover_import_transaction(
                    runtime_checkpoint,
                    projection_checkpoint,
                    allow_projection=self._controller_native_alive(),
                    token=token,
                )
                token.mutation_started = False
                raise
        except _ImportLifecycleClosed:
            return self._abort_closed_transaction(
                token, runtime_checkpoint, projection_checkpoint
            )
        except AudioImportFailure as error:
            if not self._transaction_open(token):
                return self._abort_closed_transaction(
                    token, runtime_checkpoint, projection_checkpoint
                )
            self._present_import_warning(error.title, error.user_message)
            if not self._transaction_open(token):
                return self._abort_closed_transaction(
                    token, runtime_checkpoint, projection_checkpoint
                )
            return self._finish_import_failure(
                command,
                error.reason,
                token=token,
                clear_runtime=error.clear_runtime,
            )
        except BaseException as error:
            if not self._transaction_open(token):
                return self._abort_closed_transaction(
                    token, runtime_checkpoint, projection_checkpoint
                )
            reason = self._bounded_error_text(error)
            self._log("error", f"audio import[{command.import_id}] failed: {reason}")
            if not self._transaction_open(token):
                return self._abort_closed_transaction(
                    token, runtime_checkpoint, projection_checkpoint
                )
            self._present_import_warning(
                "提示", "导入音频失败，请重新选择音频文件。"
            )
            if not self._transaction_open(token):
                return self._abort_closed_transaction(
                    token, runtime_checkpoint, projection_checkpoint
                )
            if (
                token.mutation_started
                and transaction_recovery is None
                and checkpoints_ready
            ):
                transaction_recovery = self._recover_import_transaction(
                    runtime_checkpoint,
                    projection_checkpoint,
                    allow_projection=self._controller_native_alive(),
                    token=token,
                )
                token.mutation_started = False
            if not self._transaction_open(token):
                self._retire_aborted_import(token)
                self._release_import_transaction(token)
                return False
            clear_runtime = not staging_completed
            if staging_completed and not checkpoints_ready:
                clear_runtime = False
            if transaction_recovery is not None:
                clear_runtime = False
            return self._finish_import_failure(
                command, reason, token=token, clear_runtime=clear_runtime
            )

        try:
            self._require_transaction_open(token)
            return self._publish_import_terminal(
                self.bus.events.imported_audio_ready,
                event,
                command.import_id,
                token=token,
            )
        except _ImportLifecycleClosed:
            return self._abort_closed_transaction(
                token, runtime_checkpoint, projection_checkpoint
            )
        finally:
            self._release_import_transaction(token)

    @pyqtSlot(object)
    def handle_cancel_imported_audio_requested(
        self, command: CancelImportedAudioRequested
    ) -> bool:
        if type(command) is not CancelImportedAudioRequested:
            return False
        if not self._owner_thread_is_current():
            return False
        if not self._controller_open():
            return False
        if (
            self.model.active_import_id != command.import_id
            or self.model.active_import_workflow_generation
            != command.workflow_generation
            or self.model.import_cancel_pending
        ):
            return False
        try:
            identity = self._workflow_identity()
        except _WorkflowIdentityProviderFailure:
            return False
        if not self._controller_open():
            return False
        if identity is not None and (
            identity.get("import_id") != command.import_id
            or identity.get("phase") not in {"IMPORTING", "CANCELLING"}
            or (
                identity.get("workflow_generation") is not None
                and identity.get("workflow_generation")
                != command.workflow_generation
            )
        ):
            return False
        with self._state_lock:
            if (
                not self._controller_native_alive()
                or not self._import_terminal_dispatch_active
            ):
                return False
            self.model.request_import_cancel(command.reason)
        return True

    def _accept_import_identity(self, command: LoadImportedAudioRequested) -> bool:
        identity = self._workflow_identity()
        if identity is None:
            return True
        accepted = (
            identity.get("import_id") == command.import_id
            and identity.get("phase") in {"IMPORTING", "CANCELLING"}
            and (
                identity.get("workflow_generation") is None
                or identity.get("workflow_generation")
                == command.workflow_generation
            )
        )
        if not accepted:
            self._log("warning", f"ignored stale audio import {command.import_id}")
        return accepted

    def _import_identity_is_current(
        self, command: LoadImportedAudioRequested
    ) -> bool:
        if self.model.active_import_id != command.import_id:
            return False
        try:
            identity = self._workflow_identity()
        except _WorkflowIdentityProviderFailure:
            return False
        if identity is None:
            return True
        return (
            identity.get("import_id") == command.import_id
            and identity.get("phase") == "IMPORTING"
            and (
                identity.get("workflow_generation") is None
                or identity.get("workflow_generation")
                == command.workflow_generation
            )
        )

    @staticmethod
    def _ready_import_event(
        command: LoadImportedAudioRequested, stage: ImportedAudioStage
    ) -> ImportedAudioReady:
        reference_snapshot = None
        if stage.reference is not None:
            reference_snapshot = {
                "sample_rate": getattr(
                    stage.reference, "sample_rate", stage.sample_rate
                ),
                "total_time": getattr(stage.reference, "total_time", None),
            }
        return ImportedAudioReady(
            command.import_id,
            {
                "record_id": stage.file_path,
                "file_path": stage.file_path,
                "mode": stage.mode,
                "sample_rate": stage.sample_rate,
                "sample_count": stage.sample_count,
                "channel_count": int(stage.audio_multi.shape[1]),
            },
            reference_snapshot,
        )

    @staticmethod
    def _runtime_attribute_checkpoint(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
        missing = object()
        return {
            name: getattr(owner, name, missing)
            for name in names
        } | {"__missing__": missing}

    def _capture_import_runtime(self) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = self.runtime
        data_struct = getattr(runtime, "data_struct", None)
        runtime_state = self._runtime_attribute_checkpoint(
            runtime, ("recorded_path", "recorded_signal_info")
        )
        data_state = self._runtime_attribute_checkpoint(
            data_struct,
            (
                "store_wave_data_multi",
                "store_wave_data",
                "sample_rate",
                "audio_lenth",
                "stimulus_data",
                "stimulus_info",
                "alignment_sample_count",
                "wav_calibration_metadata",
                "wav_calibration_metadata_authoritative",
                "wav_calibration_warning_shown",
            ),
        )
        return runtime_state, data_state

    def _restore_attributes(
        self, owner: Any, state: Mapping[str, Any], *, surface: str
    ) -> tuple[str, ...]:
        if owner is None:
            return (f"{surface}:owner-unavailable",)
        missing = state["__missing__"]
        failures = []
        for name, value in state.items():
            if name == "__missing__":
                continue
            try:
                if value is missing:
                    delattr(owner, name)
                else:
                    setattr(owner, name, value)
            except AttributeError:
                if value is not missing:
                    failures.append(f"{surface}.{name}")
            except BaseException as error:
                failures.append(f"{surface}.{name}")
                self._log(
                    "error",
                    f"audio import {surface}.{name} restore failed: "
                    f"{self._bounded_error_text(error)}",
                )
                continue
            try:
                if value is missing:
                    absent = object()
                    restored = getattr(owner, name, absent) is absent
                else:
                    observed = getattr(owner, name)
                    restored = (
                        observed == value
                        if type(value) in {bool, int, float, str, bytes}
                        else observed is value
                    )
                    if type(restored) is not bool:
                        restored = False
                if not restored:
                    failures.append(f"{surface}.{name}")
            except BaseException as error:
                failures.append(f"{surface}.{name}")
                self._log(
                    "error",
                    f"audio import {surface}.{name} restore verification failed: "
                    f"{self._bounded_error_text(error)}",
                )
        return tuple(failures)

    def _restore_import_runtime(
        self, checkpoint: tuple[dict[str, Any], dict[str, Any]]
    ) -> tuple[str, ...]:
        runtime_state, data_state = checkpoint
        failures = list(
            self._restore_attributes(self.runtime, runtime_state, surface="runtime")
        )
        try:
            data_struct = getattr(self.runtime, "data_struct", None)
        except BaseException as error:
            self._log(
                "error",
                "audio import data_struct lookup failed: "
                f"{self._bounded_error_text(error)}",
            )
            failures.append("runtime.data_struct")
            data_struct = None
        failures.extend(
            self._restore_attributes(data_struct, data_state, surface="data_struct")
        )
        return tuple(failures)

    def _capture_import_projection(self) -> Any:
        callback = getattr(self.view, "capture_import_projection", None)
        return callback() if callable(callback) else None

    def _restore_import_projection(
        self,
        checkpoint: Any,
        *,
        token: _ImportTransactionToken | None = None,
    ) -> tuple[str, ...]:
        failures = []
        def boundary_open() -> bool:
            if not self._controller_native_alive():
                return False
            return token is None or self._transaction_open(token)

        if not boundary_open():
            return ("projection.lifecycle-closed",)
        restore_plot = getattr(self.view, "restore_import_plot", None)
        if callable(restore_plot) and type(checkpoint) is tuple and len(checkpoint) == 2:
            try:
                if restore_plot(checkpoint[0]) is False:
                    failures.append("projection.plot")
            except BaseException as error:
                failures.append("projection.plot")
                self._log(
                    "error",
                    "audio import plot restore failed: "
                    f"{self._bounded_error_text(error)}",
                )
            if not boundary_open():
                failures.append("projection.lifecycle-closed")
                return tuple(failures)
            try:
                if self.view.set_import_data_enabled(checkpoint[1]) is False:
                    failures.append("projection.enabled")
            except BaseException as error:
                failures.append("projection.enabled")
                self._log(
                    "error",
                    "audio import enabled restore failed: "
                    f"{self._bounded_error_text(error)}",
                )
            if not boundary_open():
                failures.append("projection.lifecycle-closed")
            return tuple(failures)
        callback = getattr(self.view, "restore_import_projection", None)
        if callable(callback) and boundary_open():
            try:
                if callback(checkpoint) is False:
                    failures.append("projection")
            except BaseException as error:
                failures.append("projection")
                self._log(
                    "error",
                    "audio import projection restore failed: "
                    f"{self._bounded_error_text(error)}",
                )
            if not boundary_open():
                failures.append("projection.lifecycle-closed")
        return tuple(failures)

    def _recover_import_transaction(
        self,
        runtime_checkpoint: tuple[dict[str, Any], dict[str, Any]],
        projection_checkpoint: Any,
        *,
        allow_projection: bool = True,
        fallback_to_empty: bool = True,
        token: _ImportTransactionToken | None = None,
    ) -> ImportTransactionRecovery:
        failures = list(self._restore_import_runtime(runtime_checkpoint))
        if allow_projection and self._controller_native_alive():
            failures.extend(
                self._restore_import_projection(
                    projection_checkpoint, token=token
                )
            )
        if token is not None and not self._transaction_open(token):
            runtime_failed = any(
                name.startswith("runtime.")
                or name.startswith("data_struct.")
                for name in failures
            )
            return ImportTransactionRecovery(
                not runtime_failed,
                False,
                failures=tuple(failures),
            )
        if not failures:
            self.model.set_import_consistency(True)
            return ImportTransactionRecovery(True, True)
        if not fallback_to_empty:
            reason = ", ".join(failures)
            self.model.set_import_consistency(False, reason)
            return ImportTransactionRecovery(
                False,
                False,
                consistent=False,
                failures=tuple(failures),
            )
        clear_failures = self._clear_import_runtime(
            allow_projection=allow_projection and self._controller_native_alive(),
            token=token,
        )
        if token is not None and not self._transaction_open(token):
            return ImportTransactionRecovery(
                False,
                False,
                converged_to_empty=not clear_failures,
                failures=tuple((*failures, *clear_failures)),
            )
        if not clear_failures:
            self.model.set_import_consistency(True)
            return ImportTransactionRecovery(
                False,
                False,
                converged_to_empty=True,
                failures=tuple(failures),
            )
        all_failures = tuple((*failures, *clear_failures))
        reason = ", ".join(all_failures)
        self.model.set_import_consistency(False, reason)
        self._log("error", f"audio import state is inconsistent: {reason}")
        return ImportTransactionRecovery(
            False,
            False,
            consistent=False,
            failures=all_failures,
        )

    def _commit_import_stage(
        self,
        stage: ImportedAudioStage,
        *,
        token: _ImportTransactionToken,
    ) -> None:
        runtime = self.runtime
        data_struct = getattr(runtime, "data_struct", None)
        self._require_transaction_open(token)
        if data_struct is None:
            raise RuntimeError("analysis runtime data is unavailable")
        runtime.recorded_path = stage.file_path
        self._require_transaction_open(token)
        runtime.recorded_signal_info = {
            "file_path": stage.file_path,
            "barcode": None,
            "labels": "not_labeled",
        }
        self._require_transaction_open(token)
        data_struct.store_wave_data_multi = stage.audio_multi
        self._require_transaction_open(token)
        data_struct.store_wave_data = stage.audio_mono
        self._require_transaction_open(token)
        data_struct.sample_rate = stage.sample_rate
        data_struct.audio_lenth = stage.sample_count
        self._require_transaction_open(token)
        data_struct.wav_calibration_metadata = mutable_import_snapshot(
            stage.calibration_metadata
        )
        self._require_transaction_open(token)
        data_struct.wav_calibration_metadata_authoritative = True
        self._require_transaction_open(token)
        data_struct.wav_calibration_warning_shown = False
        self._require_transaction_open(token)
        if stage.reference is None:
            data_struct.stimulus_data = None
            self._require_transaction_open(token)
            data_struct.stimulus_info = None
            self._require_transaction_open(token)
            if hasattr(data_struct, "alignment_sample_count"):
                delattr(data_struct, "alignment_sample_count")
                self._require_transaction_open(token)
            return
        data_struct.stimulus_data = getattr(stage.reference, "stimulus_data", None)
        self._require_transaction_open(token)
        data_struct.stimulus_info = mutable_import_snapshot(
            getattr(stage.reference, "stimulus_info", None)
        )
        self._require_transaction_open(token)
        if hasattr(data_struct, "alignment_sample_count"):
            delattr(data_struct, "alignment_sample_count")
            self._require_transaction_open(token)
        alignment_present = getattr(
            stage.reference, "alignment_sample_count_present", None
        )
        if type(alignment_present) is not bool:
            alignment_present = hasattr(stage.reference, "alignment_sample_count")
        alignment = getattr(stage.reference, "alignment_sample_count", None)
        if alignment_present and alignment is not None:
            data_struct.alignment_sample_count = alignment
            self._require_transaction_open(token)
        data_struct.sample_rate = stage.sample_rate
        self._require_transaction_open(token)

    def _project_import_stage(
        self,
        stage: ImportedAudioStage,
        *,
        token: _ImportTransactionToken,
    ) -> None:
        if stage.calibration_metadata is None:
            self._present_import_warning(
                "提示", "该音频文件未包含有效校准数据，分析结果仅供参考。"
            )
            self._require_transaction_open(token)
            data_struct = self.runtime.data_struct
            self._require_transaction_open(token)
            data_struct.wav_calibration_warning_shown = True
            self._require_transaction_open(token)
        self.view.show_imported_audio(stage.audio_multi, stage.sample_rate)
        self._require_transaction_open(token)
        if self.view.set_import_data_enabled(True) is False:
            raise RuntimeError("import data projection was rejected")
        self._require_transaction_open(token)

    def _clear_import_runtime(
        self,
        *,
        allow_projection: bool = True,
        token: _ImportTransactionToken | None = None,
    ) -> tuple[str, ...]:
        runtime = self.runtime
        failures = []
        try:
            data_struct = getattr(runtime, "data_struct", None)
        except BaseException as error:
            data_struct = None
            failures.append("runtime.data_struct")
            self._log("error", f"audio import data_struct clear lookup failed: {self._bounded_error_text(error)}")
        if data_struct is not None:
            for name in (
                "store_wave_data",
                "store_wave_data_multi",
                "sample_rate",
                "audio_lenth",
                "stimulus_data",
                "stimulus_info",
                "wav_calibration_metadata",
            ):
                try:
                    setattr(data_struct, name, None)
                except BaseException as error:
                    failures.append(f"data_struct.{name}")
                    self._log("error", f"audio import data_struct.{name} clear failed: {self._bounded_error_text(error)}")
                    continue
                try:
                    if getattr(data_struct, name) is not None:
                        failures.append(f"data_struct.{name}")
                except BaseException as error:
                    failures.append(f"data_struct.{name}")
                    self._log("error", f"audio import data_struct.{name} clear verification failed: {self._bounded_error_text(error)}")
            for name, value in (
                ("wav_calibration_metadata_authoritative", False),
                ("wav_calibration_warning_shown", False),
            ):
                try:
                    setattr(data_struct, name, value)
                except BaseException as error:
                    failures.append(f"data_struct.{name}")
                    self._log("error", f"audio import data_struct.{name} clear failed: {self._bounded_error_text(error)}")
                    continue
                try:
                    if getattr(data_struct, name) is not value:
                        failures.append(f"data_struct.{name}")
                except BaseException as error:
                    failures.append(f"data_struct.{name}")
                    self._log("error", f"audio import data_struct.{name} clear verification failed: {self._bounded_error_text(error)}")
            try:
                delattr(data_struct, "alignment_sample_count")
            except AttributeError:
                pass
            except BaseException as error:
                failures.append("data_struct.alignment_sample_count")
                self._log("error", f"audio import alignment clear failed: {self._bounded_error_text(error)}")
        for name in ("recorded_path", "recorded_signal_info"):
            try:
                setattr(runtime, name, None)
            except BaseException as error:
                failures.append(f"runtime.{name}")
                self._log("error", f"audio import runtime.{name} clear failed: {self._bounded_error_text(error)}")
                continue
            try:
                if getattr(runtime, name) is not None:
                    failures.append(f"runtime.{name}")
            except BaseException as error:
                failures.append(f"runtime.{name}")
                self._log("error", f"audio import runtime.{name} clear verification failed: {self._bounded_error_text(error)}")
        clear_projection = getattr(self.view, "clear_import_projection", None)
        if (
            allow_projection
            and self._controller_native_alive()
            and (token is None or self._transaction_open(token))
            and callable(clear_projection)
        ):
            try:
                if clear_projection() is False:
                    failures.append("projection.plot")
            except BaseException as error:
                failures.append("projection.plot")
                self._log("error", f"audio import plot clear failed: {self._bounded_error_text(error)}")
        projection_open = bool(
            allow_projection
            and self._controller_native_alive()
            and (token is None or self._transaction_open(token))
        )
        enabled = getattr(self.view, "set_import_data_enabled", None)
        if (
            projection_open
            and callable(enabled)
        ):
            try:
                if enabled(False) is False:
                    failures.append("projection.enabled")
            except BaseException as error:
                failures.append("projection.enabled")
                self._log("error", f"audio import enabled clear failed: {self._bounded_error_text(error)}")
        return tuple(failures)

    def _present_import_warning(self, title: str, text: str) -> None:
        callback = getattr(self.view, "present_import_warning", None)
        if not callable(callback):
            return
        try:
            callback(title, text)
        except BaseException as error:
            self._log(
                "error", f"audio import warning presentation failed: {self._bounded_error_text(error)}"
            )

    def _finish_import_failure(
        self,
        command: LoadImportedAudioRequested,
        reason: str,
        *,
        token: _ImportTransactionToken,
        clear_runtime: bool = True,
    ) -> bool:
        try:
            self._require_transaction_open(token)
            if clear_runtime:
                failures = self._clear_import_runtime(token=token)
                self._require_transaction_open(token)
                if failures:
                    self.model.set_import_consistency(False, ", ".join(failures))
                else:
                    self.model.set_import_consistency(True)
                self._require_transaction_open(token)
            event = ImportedAudioFailed(
                command.import_id, str(reason or "audio import failed")
            )
            self._require_transaction_open(token)
            self._publish_import_terminal(
                self.bus.events.imported_audio_failed,
                event,
                command.import_id,
                token=token,
            )
            return False
        except _ImportLifecycleClosed:
            return self._abort_closed_transaction(token, None, None)
        finally:
            self._release_import_transaction(token)

    @property
    def pending_import_terminal_identity(self) -> str | None:
        pending = self._pending_import_terminal
        return None if pending is None else pending.import_id

    @property
    def import_terminal_retry_timer(self) -> QTimer | None:
        timer = self._import_terminal_retry_timer
        return timer if self._native_alive(timer) else None

    @property
    def import_terminal_retry_delay_ms(self) -> int:
        return self._import_terminal_retry_delay_ms

    @property
    def import_terminal_retry_max_delay_ms(self) -> int:
        return self._import_terminal_retry_max_delay_ms

    @staticmethod
    def _import_terminal_delivery_id(event: Any, import_id: str) -> tuple[str, str]:
        return type(event).__name__, import_id

    def _abandon_import_delivery(
        self, delivery_id: tuple[str, str], reason: str = "recording-disconnect"
    ) -> bool:
        abandon = getattr(self.bus, "abandon_import_terminal", None)
        if not callable(abandon):
            return False
        try:
            return abandon(delivery_id, reason) is True
        except BaseException:
            return False

    def _abandon_pending_import_terminal(
        self,
        delivery_id: tuple[str, str],
        *,
        reason: str,
    ) -> None:
        self._abandon_import_delivery(delivery_id, reason)
        self._safe_timer_stop()
        self._pending_import_terminal = None
        self._import_terminal_lifecycle.complete()
        self._import_terminal_retry_attempt = 0
        self._import_terminal_retry_delay_ms = 0

    def _deliver_import_terminal(
        self,
        signal: Any,
        event: Any,
        import_id: str,
        *,
        token: _ImportTransactionToken | None = None,
    ) -> bool:
        if not self._owner_thread_is_current():
            return False
        delivery_id = self._import_terminal_delivery_id(event, import_id)
        attempt = self._begin_import_terminal_attempt(delivery_id)
        if attempt is None:
            return False
        try:
            return self._deliver_import_terminal_attempt(
                signal,
                event,
                import_id,
                attempt=attempt,
                token=token,
            )
        finally:
            self._end_import_terminal_attempt(attempt)

    def _deliver_import_terminal_attempt(
        self,
        signal: Any,
        event: Any,
        import_id: str,
        *,
        attempt: _ImportTerminalAttempt,
        token: _ImportTransactionToken | None = None,
    ) -> bool:
        if token is not None:
            self._require_transaction_open(token)
            token.terminal_started = True
        elif not self._controller_open():
            return False
        delivery_id = self._import_terminal_delivery_id(event, import_id)
        has_recipients = getattr(self.bus, "has_import_terminal_recipients", None)
        dispatcher = getattr(self.bus, "deliver_import_terminal", None)
        canonical_acknowledged = False
        if callable(dispatcher):
            try:
                recipients_available = (
                    not callable(has_recipients) or has_recipients()
                )
            except BaseException as error:
                recipients_available = False
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] recipient lookup failed: "
                    f"{self._bounded_error_text(error)}",
                )
            if token is not None:
                self._require_transaction_open(token)
            elif not self._controller_open():
                return False
            if not recipients_available:
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] has no critical recipient",
                )
                if token is not None:
                    self._require_transaction_open(token)
                return False
            self._import_terminal_lifecycle.begin_delivery(delivery_id, attempt)
            try:
                accepted = dispatcher(
                    delivery_id, event
                ) is True
            except BaseException as error:
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] delivery failed: "
                    f"{self._bounded_error_text(error)}",
                )
                self._import_terminal_lifecycle.end_delivery(
                    delivery_id, attempt, acknowledged=False
                )
                return False
            if accepted:
                canonical_acknowledged = True
                if token is not None:
                    # The canonical dispatcher ACK is the irreversible import
                    # commit point.  Record it before any lifecycle recheck.
                    token.terminal_acknowledged = True
                    token.mutation_started = False
            self._import_terminal_lifecycle.end_delivery(
                delivery_id,
                attempt,
                acknowledged=canonical_acknowledged,
            )
            if not canonical_acknowledged:
                if token is not None and not self._transaction_open(token):
                    self._abandon_import_delivery(delivery_id)
                    raise _ImportLifecycleClosed(
                        "import lifecycle closed during terminal delivery"
                    )
                if token is None and not self._controller_open():
                    self._abandon_import_delivery(delivery_id)
                    self._pending_import_terminal = None
                    return False
                return False
            if token is not None and not self._transaction_open(token):
                return True
            if token is None and not self._controller_open():
                return True
        else:
            canonical_acknowledged = True
            if token is not None:
                token.terminal_acknowledged = True
                token.mutation_started = False
        if canonical_acknowledged:
            try:
                signal.emit(event)
            except BaseException as error:
                # Raw Qt observers are compatibility-only and cannot veto the
                # canonical Workflow acknowledgement.
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] observer failed: "
                    f"{self._bounded_error_text(error)}",
                )
            return True
        return False

    def _publish_import_terminal(
        self,
        signal: Any,
        event: Any,
        import_id: str,
        *,
        token: _ImportTransactionToken | None = None,
    ) -> bool:
        if token is not None:
            self._require_transaction_open(token)
        elif not self._controller_open():
            return False
        if self._pending_import_terminal is not None:
            return False
        if not self._deliver_import_terminal(
            signal, event, import_id, token=token
        ):
            delivery_id = self._import_terminal_delivery_id(event, import_id)
            if token is not None:
                if not self._transaction_open(token):
                    self._abandon_import_delivery(delivery_id)
                self._require_transaction_open(token)
            permanently_rejected = self._import_terminal_is_permanently_rejected(
                event, import_id
            )
            if token is not None and not self._transaction_open(token):
                self._abandon_import_delivery(delivery_id)
                raise _ImportLifecycleClosed(
                    "import lifecycle closed during terminal classification"
                )
            if token is None and not self._controller_open():
                self._abandon_import_delivery(delivery_id)
                return False
            if permanently_rejected:
                self._finish_permanently_rejected_import_terminal(import_id)
                return False
            self._pending_import_terminal = _PendingImportTerminal(
                import_id, signal, event
            )
            self._import_terminal_lifecycle.set_pending(
                self._import_terminal_delivery_id(event, import_id)
            )
            if token is not None:
                self._require_transaction_open(token)
            if not self._schedule_import_terminal_retry(token=token):
                if token is not None and not self._transaction_open(token):
                    self._abandon_pending_import_terminal(
                        delivery_id, reason="recording-disconnect"
                    )
                    raise _ImportLifecycleClosed(
                        "import lifecycle closed before terminal retry"
                    )
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] retry driver unavailable",
                )
            return False
        if token is not None and not token.terminal_acknowledged:
            self._require_transaction_open(token)
        self.model.retire_import(import_id)
        return True

    def _schedule_import_terminal_retry(
        self, *, token: _ImportTransactionToken | None = None
    ) -> bool:
        if not self._owner_thread_is_current():
            return self._request_import_terminal_retry_driver()
        if token is not None and not self._transaction_open(token):
            return False
        if (
            not self._controller_open()
            or self._pending_import_terminal is None
        ):
            return False
        timer = self._ensure_import_terminal_retry_timer()
        if timer is None:
            return self._request_import_terminal_retry_driver()
        try:
            if timer.isActive():
                return False
        except (RuntimeError, TypeError):
            replacement = self._ensure_import_terminal_retry_timer()
            if replacement is None or replacement is timer:
                return self._request_import_terminal_retry_driver()
            timer = replacement
            try:
                if timer.isActive():
                    return False
            except (RuntimeError, TypeError):
                return self._request_import_terminal_retry_driver()
        exponent = min(self._import_terminal_retry_attempt, 30)
        delay = min(
            self._import_terminal_retry_base_delay_ms * (2 ** exponent),
            self._import_terminal_retry_max_delay_ms,
        )
        self._import_terminal_retry_attempt += 1
        self._import_terminal_retry_delay_ms = delay
        if token is not None and not self._transaction_open(token):
            return False
        if token is None and not self._controller_open():
            return False
        try:
            timer.start(delay)
        except (RuntimeError, TypeError):
            replacement = self._ensure_import_terminal_retry_timer()
            if replacement is None or replacement is timer:
                return self._request_import_terminal_retry_driver()
            try:
                replacement.start(delay)
            except (RuntimeError, TypeError):
                return self._request_import_terminal_retry_driver()
            timer = replacement
        if not self._native_alive(timer):
            return self._request_import_terminal_retry_driver()
        return True

    @pyqtSlot()
    def handle_import_terminal_retry_timeout(self) -> bool:
        if not self._owner_thread_is_current():
            return False
        self._safe_timer_stop()
        pending = self._pending_import_terminal
        if pending is None or not self._controller_open():
            return False
        return self.retry_pending_import_terminal(pending.import_id)

    def retry_pending_import_terminal(self, import_id: str) -> bool:
        if not self._owner_thread_is_current():
            return False
        pending = self._pending_import_terminal
        if pending is None or pending.import_id != import_id:
            return False
        if not self._controller_open():
            return False
        self._safe_timer_stop()
        if not self._deliver_import_terminal(
            pending.signal, pending.event, pending.import_id
        ):
            if self._import_terminal_attempt_is_active():
                return False
            if not self._controller_open():
                return False
            permanently_rejected = self._import_terminal_is_permanently_rejected(
                pending.event, pending.import_id
            )
            if not self._controller_open():
                return False
            if permanently_rejected:
                self._finish_permanently_rejected_import_terminal(import_id)
                return False
            if not self._schedule_import_terminal_retry():
                self._log(
                    "error",
                    f"audio import terminal[{import_id}] retry driver unavailable",
                )
            return False
        self._pending_import_terminal = None
        self._import_terminal_lifecycle.complete()
        self._import_terminal_retry_attempt = 0
        self._import_terminal_retry_delay_ms = 0
        self.model.retire_import(import_id)
        return True

    def _import_terminal_is_permanently_rejected(
        self, event: Any, import_id: str
    ) -> bool:
        classify = getattr(
            self.bus, "classify_import_terminal_delivery", None
        )
        if not callable(classify):
            return False
        try:
            result = classify(
                self._import_terminal_delivery_id(event, import_id), event
            )
        except BaseException:
            return False
        return result is ImportTerminalRecipientResult.PERMANENT_REJECT

    def _finish_permanently_rejected_import_terminal(
        self, import_id: str
    ) -> None:
        self._safe_timer_stop()
        self._pending_import_terminal = None
        self._import_terminal_lifecycle.complete()
        self._import_terminal_retry_attempt = 0
        self._import_terminal_retry_delay_ms = 0
        self.model.retire_import(import_id)

    def disconnect(self, _lifecycle_request=None) -> bool:
        if not self._controller_native_alive():
            return type(_lifecycle_request) is ResourceLifecycleRequested
        with self._state_lock:
            if not self._import_terminal_dispatch_active:
                return type(_lifecycle_request) is ResourceLifecycleRequested
            self._import_terminal_dispatch_active = False
            self._lifecycle_generation += 1
            active = self._active_import_transaction
            if active is not None:
                active.invalidated = True
        self._safe_timer_stop()
        pending = self._pending_import_terminal
        if pending is not None:
            self._import_terminal_lifecycle.close()
            self._pending_import_terminal = None
        if (
            active is not None
            and active.model_begun
            and not active.mutation_started
            and not active.terminal_started
        ):
            self._retire_aborted_import(active)
        return True
