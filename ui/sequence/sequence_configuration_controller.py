"""Configuration loading, selection, and runtime-reference orchestration."""

from __future__ import annotations

import math
import os
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

from base.acquisition_recording_defaults import normalize_record_only_detail
from base.audio_sample_rate import resolve_duplex_sample_rate, resolve_input_sample_rate
from base.load_config import LoadUiConfig, PathTransactionCoordinator
from base.log_manager import LogManager
from base.stimulus_resolver import set_data_struct_analysis_reference_signal
from consts import error_code
from consts.running_consts import SEQUENCE_CONFIG_REGISTRY_PATH
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.sequence_configuration_analysis_flags import (
    DataStructAnalysisFlagProjectionPort,
    SequenceAnalysisFlagProjectionService,
)
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_configuration_view import SequenceConfigurationView
from ui.sequence.sequence_messages import ConfigurationChanged, ConfigurationSnapshot


_CURRENT_CONFIGURATION_PATH = object()


class _ReconcileRegistryAfterTransaction:
    pass


_RECONCILE_REGISTRY_AFTER_TRANSACTION = _ReconcileRegistryAfterTransaction()

_PROJECTION_SURFACE_ORDER = (
    "registry",
    "configuration",
    "channels",
    "analysis",
    "mode_availability",
    "runtime",
    "plot_import",
    "view",
)
_PERSISTENCE_SURFACE_ORDER = (
    "registry",
    "configuration_file",
)


def _canonical_json_semantic_token(value: Any) -> Any:
    """Return an exact-type JSON token without consulting live-object equality.

    Object member order is ignored. All NaNs share one token, infinities keep
    their sign, and finite floats use ``float.hex()`` so ``-0.0`` remains
    distinct from ``0.0``. Non-JSON values and non-string object keys fail
    closed.
    """
    value_type = type(value)
    if value_type is type(None):
        return ("null",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        if math.isnan(value):
            representation = "nan"
        elif math.isinf(value):
            representation = "+inf" if value > 0 else "-inf"
        else:
            representation = value.hex()
        return ("float", representation)
    if value_type is str:
        return ("str", value)
    if value_type is list:
        return (
            "list",
            tuple(_canonical_json_semantic_token(item) for item in value),
        )
    if value_type is dict:
        members = []
        for key, item in dict.items(value):
            if type(key) is not str:
                raise TypeError(
                    "JSON semantic objects require exact string keys, got "
                    f"{type(key).__name__}"
                )
            members.append(
                (key, _canonical_json_semantic_token(item))
            )
        members.sort(key=lambda member: member[0])
        return ("object", tuple(members))
    raise TypeError(
        "unsupported JSON semantic value: "
        f"{value_type.__name__}"
    )


def _durable_ownership_token(value: Any) -> Any:
    """Tokenize durable state without consulting coercive object equality."""
    if (
        type(value) is tuple
        and len(value) == 2
        and type(value[0]) is bool
        and type(value[1]) is bytes
    ):
        return ("file-bytes", value[0], value[1])
    return ("json", _canonical_json_semantic_token(value))


def _durable_ownership_tokens_equal(left: Any, right: Any) -> bool:
    """Compare only tokens produced by :func:`_durable_ownership_token`."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(
            _durable_ownership_tokens_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if type(left) in {bool, int, str, bytes}:
        return left == right
    return False


def _safe_exception_description(error: BaseException) -> str:
    try:
        detail = str(error)
    except BaseException:
        detail = "<unprintable exception>"
    return f"{type(error).__name__}: {detail}"


def _add_exception_note_safely(error: BaseException, note: str) -> None:
    try:
        BaseException.add_note(error, note)
    except BaseException:
        # Diagnostics are best effort and must never replace the primary error.
        return


@dataclass(frozen=True, slots=True)
class _RecoveryFailure:
    operation: str
    error: BaseException
    traceback: Any


class _RecoveryFailureAggregator:
    """Select one recovery failure without losing later cleanup attempts.

    Interruptions always outrank ordinary exceptions. Within either category,
    the earliest eligible failure stays primary. Ordinary diagnostic failures
    are retained as notes but are not independently raised.
    """

    def __init__(
        self,
        primary_error: BaseException | None = None,
        *,
        operation: str = "primary operation",
    ) -> None:
        self._records: list[_RecoveryFailure] = []
        self._primary: _RecoveryFailure | None = None
        if primary_error is not None:
            self.capture(operation, primary_error)

    @staticmethod
    def _is_interruption(error: BaseException) -> bool:
        return not isinstance(error, Exception)

    def _add_record_note(self, record: _RecoveryFailure) -> None:
        primary = self._primary
        if primary is None or record.error is primary.error:
            return
        _add_exception_note_safely(
            primary.error,
            f"{record.operation} also failed: "
            f"{_safe_exception_description(record.error)}",
        )

    def capture(
        self,
        operation: str,
        error: BaseException,
    ) -> None:
        record = _RecoveryFailure(operation, error, error.__traceback__)
        previous_records = tuple(self._records)
        self._records.append(record)
        interruption = self._is_interruption(error)
        if self._primary is None:
            self._primary = record
            for previous in previous_records:
                self._add_record_note(previous)
            return
        if (
            interruption
            and not self._is_interruption(self._primary.error)
        ):
            self._primary = record
            for previous in previous_records:
                self._add_record_note(previous)
            return
        self._add_record_note(record)

    def warning(self, logger: Any, message: str, *, operation: str) -> None:
        try:
            logger.warning(message)
        except BaseException as error:
            self.capture(operation, error)

    @property
    def primary_error(self) -> BaseException | None:
        return self._primary.error if self._primary is not None else None

    @property
    def failure_count(self) -> int:
        return len(self._records)

    @property
    def has_failures(self) -> bool:
        return bool(self._records)

    @property
    def records(self) -> tuple[_RecoveryFailure, ...]:
        return tuple(self._records)

    @property
    def has_interruption(self) -> bool:
        return bool(
            self._primary is not None
            and self._is_interruption(self._primary.error)
        )

    def raise_if_selected(self) -> None:
        primary = self._primary
        if primary is None:
            return
        primary.error.__traceback__ = primary.traceback
        raise primary.error.with_traceback(primary.traceback)

    def raise_if_interrupted(self) -> None:
        if self.has_interruption:
            self.raise_if_selected()


@dataclass(frozen=True, slots=True)
class _RuntimePreparation:
    failure: str | None = None
    regenerated: bool = False
    verified_surfaces: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _CapturedRuntimeField:
    existed: bool
    value: Any


@dataclass(frozen=True, slots=True)
class _ActivePathSemanticCheckpoint:
    using_path_present: bool
    using_config_path: Any
    selected_key: Any
    selected_path_present: bool
    selected_path: Any


@dataclass(slots=True)
class _DurableWriteCheckpoint:
    operation: str
    rollback: Callable[[Any], Any]
    state_reader: Callable[[], Any] | None
    ownership_tokenizer: Callable[[Any], Any]
    surface: str
    owned_token: Any = None
    state_captured: bool = False


@dataclass(frozen=True, slots=True)
class _UsingPathPersistenceResult:
    succeeded: bool
    committed_registry: dict[Any, Any] | None = None

    def __bool__(self) -> bool:
        return self.succeeded


@dataclass(slots=True)
class PersistenceAdapter:
    """Complete coordinated persistence protocol for one durable boundary."""

    coordinator: PathTransactionCoordinator
    transaction_key: Callable[[Any], Any]
    checkpoint_reader: Callable[[Any], Any]
    writer: Callable[[Any, Any], Any]
    checkpoint_restorer: Callable[[Any, Any], Any]
    durable_truth_reader: Callable[[Any], Any]
    semantic_reader: Callable[[Any], Any] | None = None
    ownership_tokenizer: Callable[[Any], Any] = _durable_ownership_token

    def transaction(self, target: Any):
        return _PersistenceContext(
            self.coordinator.transaction(self.transaction_key(target))
        )

    def capture(self, target: Any) -> Any:
        return self.checkpoint_reader(target)

    def write(self, payload: Any, target: Any) -> Any:
        return self.writer(payload, target)

    def conditional_restore(
        self, target: Any, checkpoint: Any, owned_token: Any
    ) -> Any:
        with self.transaction(target):
            try:
                current_token = self.ownership_token(self.capture(target))
            except Exception:
                return False
            if not _durable_ownership_tokens_equal(
                current_token, owned_token
            ):
                return False
            return self.checkpoint_restorer(target, checkpoint)

    def ownership_token(self, state: Any) -> Any:
        return self.ownership_tokenizer(state)

    def read_durable_truth(self, target: Any) -> Any:
        with self.transaction(target):
            return self.durable_truth_reader(target)

    def read_semantic_current(self, target: Any) -> Any:
        reader = self.semantic_reader or self.durable_truth_reader
        return reader(target)


@runtime_checkable
class PersistenceAdapterProtocol(Protocol):
    """Runtime-checkable boundary required for custom durable persistence."""

    coordinator: PathTransactionCoordinator

    def transaction(self, target: Any) -> Any: ...

    def capture(self, target: Any) -> Any: ...

    def write(self, payload: Any, target: Any) -> Any: ...

    def conditional_restore(
        self, target: Any, checkpoint: Any, owned_token: Any
    ) -> Any: ...

    def ownership_token(self, state: Any) -> Any: ...

    def read_durable_truth(self, target: Any) -> Any: ...

    def read_semantic_current(self, target: Any) -> Any: ...


class _PersistenceContext:
    """Delegate coordinator context management without changing its semantics."""

    def __init__(self, context: Any) -> None:
        self._context = context

    def __enter__(self) -> Any:
        return self._context.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._context.__exit__(exc_type, exc, traceback)


_STIMULUS_RUNTIME_FIELDS = (
    "stimulus_data",
    "stimulus_info",
    "sample_rate",
    "alignment_sample_count",
)
_CLEAR_DATA_RUNTIME_DEFAULTS = {
    "store_wave_data": None,
    "store_wave_data_multi": None,
    "wav_calibration_metadata": None,
    "wav_calibration_metadata_authoritative": False,
    "wav_calibration_warning_shown": False,
    "split_repeat_data": None,
    "fft_result": None,
    "stft_result": None,
}
_TRANSACTION_RUNTIME_FIELDS = tuple(_CLEAR_DATA_RUNTIME_DEFAULTS) + (
    "stimulus_data",
    "stimulus_info",
    "sample_rate",
    "alignment_sample_count",
    "audio_lenth",
)


def _copy_plain_configuration(value: Any) -> Any:
    """Copy admitted JSON-shaped configuration without object copy protocols."""
    if type(value) is dict:
        return {
            _copy_plain_configuration(key): _copy_plain_configuration(item)
            for key, item in dict.items(value)
        }
    if type(value) is list:
        return [_copy_plain_configuration(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_plain_configuration(item) for item in value)
    if type(value) in {type(None), bool, int, float, str}:
        return value
    raise TypeError(
        "configuration contains a non-JSON runtime value: "
        f"{type(value).__name__}"
    )


class _ConfigurationTransaction:
    """Journal reversible configuration projections until durable commit."""

    def __init__(
        self,
        controller: "SequenceConfigurationController",
        *,
        selection_path: Any,
        owner_token: object | None = None,
    ) -> None:
        self.controller = controller
        self._owner_token = owner_token
        self._owner_scope_active = False
        self.checkpoint = controller.model.checkpoint_configuration_state()
        self.data_struct_state = controller._capture_data_struct_state(
            controller.model.data_struct
        )
        self.stimulus_reference_ready = controller.model.stimulus_reference_ready
        self.registry = controller.model.registry
        self.registry_entries = controller.model.registry_entries
        self.view_state = controller.view.capture_configuration_state(
            selection_path=selection_path
        )
        self._projection_journal: list[
            tuple[str, Callable[[], Any] | None, bool, str | None]
        ] = []
        self._commit_callbacks: list[
            tuple[
                str,
                Callable[..., Any] | None,
                Callable[[], Any] | None,
                str | None,
            ]
        ] = []
        self._verified_surfaces: set[str] = set()
        self._durable_journal: list[_DurableWriteCheckpoint] = []
        self._durable_compensation_failures: list[str] = []
        self._durable_compensation_failure_surfaces: set[str] = set()
        self._entry_projection_failures = dict(
            controller._projection_failure_reasons
        )
        self._entry_persistence_failures = dict(
            controller._persistence_failure_reasons
        )
        self._failure_checkpoint_released = False
        controller._retain_failure_reason_checkpoints(
            self._entry_projection_failures,
            self._entry_persistence_failures,
        )
        self._failure_provenance_committed = False
        self._terminal_recovery_started = False
        self._unrestorable_projection_failure: str | None = None
        self._unrestorable_projection_surface: str | None = None
        self.import_identity_state: Any = _CURRENT_CONFIGURATION_PATH
        self.plot_state: Any = _CURRENT_CONFIGURATION_PATH

    def _merge_failure_provenance(
        self,
        category: str,
        current: dict[str, tuple[str, ...]],
        entry: dict[str, tuple[str, ...]],
    ) -> None:
        for surface, entry_reasons in entry.items():
            current_reasons = current.get(surface, ())
            current[surface] = self.controller._ordered_failure_reasons(
                category,
                surface,
                (*entry_reasons, *current_reasons),
            )

    def restore_failure_provenance(self) -> None:
        if self._failure_provenance_committed:
            return
        controller = self.controller
        self._merge_failure_provenance(
            "projection",
            controller._projection_failure_reasons,
            self._entry_projection_failures,
        )
        self._merge_failure_provenance(
            "persistence",
            controller._persistence_failure_reasons,
            self._entry_persistence_failures,
        )
        controller._sync_consistency_diagnostics()

    def commit_failure_provenance(self) -> None:
        self._failure_provenance_committed = True
        self._release_failure_checkpoint()
        self._release_owner()

    def _release_failure_checkpoint(self) -> None:
        if self._failure_checkpoint_released:
            return
        self._failure_checkpoint_released = True
        self.controller._release_failure_reason_checkpoints(
            self._entry_projection_failures,
            self._entry_persistence_failures,
        )

    def enter_owner_scope(self) -> None:
        self._owner_scope_active = True

    def _release_owner(self, *, force: bool = False) -> None:
        if self._owner_scope_active and not force:
            return
        owner_token = self._owner_token
        if owner_token is None:
            return
        self._owner_token = None
        self.controller._release_configuration_transaction_owner(owner_token)

    def finalize_owner(self) -> None:
        try:
            if (
                not self._failure_provenance_committed
                and not self._failure_checkpoint_released
            ):
                self.restore_failure_provenance()
        finally:
            self._release_failure_checkpoint()
            self._owner_scope_active = False
            self._release_owner(force=True)

    def recover_base_exception(
        self,
        error: BaseException,
        *,
        failure: str,
    ) -> None:
        if self._failure_provenance_committed or self._terminal_recovery_started:
            return
        self._terminal_recovery_started = True
        self.controller._recover_before_base_exception(
            self,
            error,
            failure=failure,
        )

    def mark_terminal_recovery_started(self) -> None:
        self._terminal_recovery_started = True

    def call_owned(
        self,
        operation: str,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            return callback(*args, **kwargs)
        except BaseException as error:
            if not isinstance(error, Exception):
                self.recover_base_exception(
                    error,
                    failure=f"{operation} was interrupted",
                )
            raise

    def capture_import_identity_state(self) -> Any:
        if self.import_identity_state is _CURRENT_CONFIGURATION_PATH:
            controller = self.controller
            try:
                controller._validate_projection_hook_pair(
                    "plot_import",
                    "retained import identity",
                    controller._clear_import_identity,
                    controller._import_identity_state_capturer,
                    controller._import_identity_state_restorer,
                )
            except Exception:
                self._unrestorable_projection_failure = (
                    "incomplete retained import identity projection hooks"
                )
                self._unrestorable_projection_surface = "plot_import"
                raise
            self.import_identity_state = (
                controller._capture_optional_projection_state(
                    "retained import identity",
                    controller._clear_import_identity,
                    controller._import_identity_state_capturer,
                )
            )
        return self.import_identity_state

    def capture_plot_state(self) -> Any:
        if self.plot_state is _CURRENT_CONFIGURATION_PATH:
            controller = self.controller
            try:
                controller._validate_projection_hook_pair(
                    "plot_import",
                    "plot presentation",
                    controller._clear_plot,
                    controller._plot_state_capturer,
                    controller._plot_state_restorer,
                )
            except Exception:
                self._unrestorable_projection_failure = (
                    "incomplete plot presentation projection hooks"
                )
                self._unrestorable_projection_surface = "plot_import"
                raise
            self.plot_state = controller._capture_optional_projection_state(
                "plot presentation",
                controller._clear_plot,
                controller._plot_state_capturer,
            )
        return self.plot_state

    def attempt(
        self,
        operation: str,
        callback: Callable[..., Any] | None,
        *args: Any,
        rollback: Callable[[], Any] | None = None,
        reject_false: bool = False,
        required_projection: bool = False,
        surface: str | None = None,
        verifies_surface: bool = True,
        **kwargs: Any,
    ) -> bool:
        if callback is None:
            return True
        # Journal before invoking: callbacks may mutate and then raise.
        self._projection_journal.append(
            (operation, rollback, required_projection, surface)
        )
        if required_projection and rollback is None:
            self._unrestorable_projection_failure = (
                f"missing rollback for required projection: {operation}"
            )
            self._unrestorable_projection_surface = surface
        succeeded = self.call_owned(
            operation,
            self.controller._invoke_callback,
            operation,
            callback,
            *args,
            reject_false=reject_false,
            **kwargs,
        )
        if succeeded and surface is not None and verifies_surface:
            self._verified_surfaces.add(surface)
        return succeeded

    def attempt_result(
        self,
        operation: str,
        callback: Callable[..., Any],
        *args: Any,
        rollback: Callable[[], Any] | None = None,
        required_projection: bool = False,
        surface: str | None = None,
        verifies_surface: bool = True,
        **kwargs: Any,
    ) -> tuple[bool, Any]:
        self._projection_journal.append(
            (operation, rollback, required_projection, surface)
        )
        try:
            result = callback(*args, **kwargs)
        except Exception as exc:
            self.controller._logger.warning(f"Failed to {operation}: {exc}")
            return False, None
        except BaseException as error:
            self.recover_base_exception(
                error,
                failure=f"{operation} was interrupted",
            )
            raise
        if surface is not None and verifies_surface:
            self._verified_surfaces.add(surface)
        return True, result

    def defer_commit(
        self,
        operation: str,
        callback: Callable[..., Any] | None,
        *,
        rollback: Callable[[], Any] | None = None,
        surface: str | None = None,
    ) -> None:
        if callback is not None:
            self._commit_callbacks.append((operation, callback, rollback, surface))

    def mark_surface_verified(self, surface: str) -> None:
        self._verified_surfaces.add(surface)

    @property
    def verified_surfaces(self) -> frozenset[str]:
        return frozenset(self._verified_surfaces)

    def attempt_durable_write(
        self,
        operation: str,
        callback: Callable[..., Any],
        *args: Any,
        rollback: Callable[[Any], Any],
        state_reader: Callable[[], Any] | None = None,
        ownership_tokenizer: Callable[[Any], Any] = _durable_ownership_token,
        surface: str = "configuration_file",
    ) -> bool:
        # Register before invoking because a writer may replace bytes then fail.
        checkpoint = _DurableWriteCheckpoint(
            operation,
            rollback,
            state_reader,
            ownership_tokenizer,
            surface,
        )
        self._durable_journal.append(checkpoint)
        write_succeeded = False
        failures = _RecoveryFailureAggregator()
        try:
            result = callback(*args)
        except BaseException as exc:
            failures.capture("Durable writer", exc)
            if isinstance(exc, Exception):
                failures.warning(
                    self.controller._logger,
                    f"Failed to {operation}: "
                    f"{_safe_exception_description(exc)}",
                    operation="Durable writer diagnostic logging",
                )
        else:
            write_succeeded = result is not False
        if state_reader is not None:
            try:
                checkpoint.owned_token = checkpoint.ownership_tokenizer(
                    state_reader()
                )
                checkpoint.state_captured = True
            except BaseException as exc:
                write_succeeded = False
                failures.capture("Durable ownership capture", exc)
                if isinstance(exc, Exception):
                    failures.warning(
                        self.controller._logger,
                        "Failed to capture durable state after "
                        f"{operation}: {_safe_exception_description(exc)}",
                        operation="Durable ownership diagnostic logging",
                    )
        if not write_succeeded:
            failures.warning(
                self.controller._logger,
                f"Failed to {operation}",
                operation="Durable failure diagnostic logging",
            )
            failures.raise_if_interrupted()
            return False
        failures.raise_if_interrupted()
        return True

    @property
    def durable_write_count(self) -> int:
        return len(self._durable_journal)

    def mark_durable_uncertain(self, operation: str, surface: str) -> None:
        self._durable_journal.append(
            _DurableWriteCheckpoint(
                operation,
                rollback=lambda _owned_state: False,
                state_reader=None,
                ownership_tokenizer=_durable_ownership_token,
                surface=surface,
                state_captured=False,
            )
        )

    def compensate_durable_writes(self) -> bool:
        consistent = True
        self._durable_compensation_failures.clear()
        self._durable_compensation_failure_surfaces.clear()
        failures = _RecoveryFailureAggregator()
        for checkpoint in reversed(self._durable_journal):
            if checkpoint.state_reader is not None and not checkpoint.state_captured:
                failure = (
                    f"{checkpoint.operation}: owned durable state is unavailable"
                )
                self._durable_compensation_failures.append(failure)
                self._durable_compensation_failure_surfaces.add(
                    checkpoint.surface
                )
                failures.warning(
                    self.controller._logger,
                    f"Failed to compensate {failure}",
                    operation=(
                        f"{checkpoint.operation} compensation diagnostic logging"
                    ),
                )
                consistent = False
                continue
            try:
                restored = checkpoint.rollback(checkpoint.owned_token)
            except BaseException as exc:
                failure = (
                    f"{checkpoint.operation}: "
                    f"{_safe_exception_description(exc)}"
                )
                self._durable_compensation_failures.append(failure)
                self._durable_compensation_failure_surfaces.add(
                    checkpoint.surface
                )
                failures.capture(
                    f"{checkpoint.operation} durable compensation",
                    exc,
                )
                failures.warning(
                    self.controller._logger,
                    f"Failed to compensate {failure}",
                    operation=(
                        f"{checkpoint.operation} compensation diagnostic logging"
                    ),
                )
                consistent = False
                continue
            if restored is False:
                failure = (
                    f"{checkpoint.operation}: restorer rejected checkpoint"
                )
                self._durable_compensation_failures.append(failure)
                self._durable_compensation_failure_surfaces.add(
                    checkpoint.surface
                )
                failures.warning(
                    self.controller._logger,
                    f"Failed to compensate {failure}",
                    operation=(
                        f"{checkpoint.operation} compensation diagnostic logging"
                    ),
                )
                consistent = False
        self.restore_failure_provenance()
        failures.raise_if_interrupted()
        return consistent

    @property
    def durable_compensation_failures(self) -> tuple[str, ...]:
        return tuple(self._durable_compensation_failures)

    @property
    def durable_surfaces(self) -> frozenset[str]:
        return frozenset(checkpoint.surface for checkpoint in self._durable_journal)

    @property
    def durable_compensation_failure_surfaces(self) -> frozenset[str]:
        return frozenset(self._durable_compensation_failure_surfaces)

    def abort(self) -> bool:
        controller = self.controller
        failures = _RecoveryFailureAggregator()
        projections_consistent = self._unrestorable_projection_failure is None
        all_restored = projections_consistent
        failed_surfaces: set[str] = set()
        if self._unrestorable_projection_failure is not None:
            failed_surfaces.add(
                self._unrestorable_projection_surface or "view"
            )

        def attempt_restoration(
            operation: str,
            callback: Callable[..., Any],
            *args: Any,
            reject_false: bool = True,
        ) -> bool:
            try:
                result = callback(*args)
            except BaseException as exc:
                failures.capture(operation, exc)
                failures.warning(
                    controller._logger,
                    f"Failed to {operation}: {_safe_exception_description(exc)}",
                    operation=f"{operation} diagnostic logging",
                )
                return False
            if reject_false and result is False:
                rejection = RuntimeError("callback rejected operation")
                failures.capture(operation, rejection)
                failures.warning(
                    controller._logger,
                    f"Failed to {operation}: callback rejected operation",
                    operation=f"{operation} diagnostic logging",
                )
                return False
            return True

        try:
            controller.model.restore_configuration_state(self.checkpoint)
            controller.model._registry = self.registry
            controller.model._using_config_path = self.checkpoint.using_config_path
            controller.model._registry_entries = self.registry_entries
            controller._restore_data_struct_state(
                controller.model.data_struct, self.data_struct_state
            )
            controller.model.stimulus_reference_ready = self.stimulus_reference_ready
        except BaseException as exc:
            failures.capture("restore runtime configuration state", exc)
            failures.warning(
                controller._logger,
                "Failed to restore runtime configuration state: "
                f"{_safe_exception_description(exc)}",
                operation="runtime configuration restoration diagnostic logging",
            )
            projections_consistent = False
            all_restored = False
            failed_surfaces.add("runtime")
        for operation, rollback, _required_projection, surface in reversed(
            self._projection_journal
        ):
            if rollback is not None:
                restored = attempt_restoration(
                    f"restore after {operation}",
                    rollback,
                )
                all_restored = restored and all_restored
                if not restored:
                    projections_consistent = False
                    failed_surfaces.add(surface or "view")
        view_restored = attempt_restoration(
            "restore configuration view state",
            controller.view.restore_configuration_state,
            self.view_state,
        )
        if not view_restored:
            projections_consistent = False
            failed_surfaces.add("view")
        all_restored = view_restored and all_restored
        if not projections_consistent:
            failure = (
                self._unrestorable_projection_failure
                or "configuration projection rollback was incomplete"
            )
            for surface in failed_surfaces or {"view"}:
                controller._record_projection_failure(
                    surface,
                    f"{surface} projection rollback was incomplete: {failure}",
                )
            failures.warning(
                controller._logger,
                f"Failed configuration projection integrity: {failure}",
                operation="projection integrity diagnostic logging",
            )
            attempt_restoration(
                "disable inconsistent configuration state",
                controller.view.set_sequence_config_available,
                False,
                reject_false=False,
            )
        try:
            self.restore_failure_provenance()
        finally:
            self._release_failure_checkpoint()
            self._release_owner()
        failures.raise_if_interrupted()
        return bool(projections_consistent and all_restored)

    def run_commit_callbacks(self) -> bool:
        for operation, callback, rollback, surface in self._commit_callbacks:
            if not self.attempt(
                operation,
                callback,
                rollback=rollback,
                reject_false=True,
                required_projection=True,
                surface=surface,
            ):
                return False
        return True


class SequenceConfigurationController(QObject):
    """Mutate the configuration model at external/configuration boundaries."""

    def __init__(
        self,
        model: SequenceConfigurationModel,
        view: SequenceConfigurationView,
        *,
        registry_loader: Callable[[], Any] | None = None,
        config_loader: Callable[[Any], tuple[Any, Any]] | None = None,
        using_path_updater: Callable[[Any], Any] | None = None,
        config_saver: Callable[[Any, Any], Any] | None = None,
        ok_code: Any = error_code.OK,
        input_sample_rate_resolver: Callable[[Any], Any] = resolve_input_sample_rate,
        duplex_sample_rate_resolver: Callable[[Any, Any], Any] = resolve_duplex_sample_rate,
        stimulus_setter: Callable[..., Any] = AnalysisModelSelect.set_data_struct_stimulus_signal,
        analysis_reference_setter: Callable[..., Any] = set_data_struct_analysis_reference_signal,
        configuration_publisher: Callable[[ConfigurationChanged], Any] | None = None,
        warning: Callable[[str, str], Any] | None = None,
        availability_changed: Callable[[], Any] | None = None,
        data_enabled_setter: Callable[[bool], Any] | None = None,
        refresh_channels: Callable[[], Any] | None = None,
        clear_plot: Callable[[], Any] | None = None,
        clear_import_identity: Callable[[], Any] | None = None,
        plot_state_capturer: Callable[[], Any] | None = None,
        plot_state_restorer: Callable[[Any], Any] | None = None,
        import_identity_state_capturer: Callable[[], Any] | None = None,
        import_identity_state_restorer: Callable[[Any], Any] | None = None,
        config_persistence_snapshotter: Callable[[Any], Any] | None = None,
        config_persistence_restorer: Callable[[Any, Any], Any] | None = None,
        using_path_persistence_snapshotter: Callable[[], Any] | None = None,
        using_path_persistence_restorer: Callable[[Any], Any] | None = None,
        path_transaction_coordinator: PathTransactionCoordinator | None = None,
        config_persistence_adapter: PersistenceAdapterProtocol | None = None,
        using_path_persistence_adapter: PersistenceAdapterProtocol | None = None,
        using_path_persistence_transaction_key: Any = (
            SEQUENCE_CONFIG_REGISTRY_PATH
        ),
        analysis_flag_projection_service: (
            SequenceAnalysisFlagProjectionService | None
        ) = None,
        analysis_config_changed: Callable[[dict[Any, Any]], Any] | None = None,
        refresh_test_mode_availability: Callable[[], Any] | None = None,
        logger: Any = None,
        path_exists: Callable[[Any], bool] = os.path.exists,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.view = view
        adapter_coordinator = next(
            (
                candidate
                for candidate in (
                    getattr(config_persistence_adapter, "coordinator", None),
                    getattr(using_path_persistence_adapter, "coordinator", None),
                )
                if isinstance(candidate, PathTransactionCoordinator)
            ),
            None,
        )
        self._path_transaction_coordinator = (
            path_transaction_coordinator
            or adapter_coordinator
            or PathTransactionCoordinator()
        )
        self._config_persistence_adapter_override = config_persistence_adapter
        self._using_path_persistence_adapter_override = (
            using_path_persistence_adapter
        )
        self._using_path_persistence_transaction_key = (
            using_path_persistence_transaction_key
        )
        self._last_config_persistence_adapter: PersistenceAdapterProtocol | None = None
        self._last_using_path_persistence_adapter: PersistenceAdapterProtocol | None = None
        self._registry_loader = (
            registry_loader or LoadUiConfig._load_sequence_config_registry
        )
        self._config_loader = config_loader or (
            lambda path: LoadUiConfig().load_sequence_config_from_json(path)
        )
        using_default_path_writer = using_path_updater is None
        using_default_config_writer = config_saver is None
        self._using_path_updater = using_path_updater or (
            lambda path: LoadUiConfig.update_using_config_path(
                path,
                coordinator=self._path_transaction_coordinator,
            )
        )
        self._config_saver = config_saver or (
            lambda config, path: LoadUiConfig.save_sequence_config_to_json(
                config,
                path,
                coordinator=self._path_transaction_coordinator,
            )
        )
        self._ok_code = ok_code
        self._resolve_input_sample_rate = input_sample_rate_resolver
        self._resolve_duplex_sample_rate = duplex_sample_rate_resolver
        self._stimulus_setter = stimulus_setter
        self._analysis_reference_setter = analysis_reference_setter
        self._publish_configuration = configuration_publisher
        self._warning = warning or view.warn
        self.view.bind_availability_callback(availability_changed)
        self.view.bind_runtime_readiness_provider(self.runtime_action_readiness)
        self._data_enabled_setter = data_enabled_setter or view.set_data_enabled
        self._refresh_channels = refresh_channels
        self._clear_plot = clear_plot
        self._clear_import_identity = clear_import_identity
        self._plot_state_capturer = plot_state_capturer
        self._plot_state_restorer = plot_state_restorer
        self._import_identity_state_capturer = import_identity_state_capturer
        self._import_identity_state_restorer = import_identity_state_restorer
        self._config_persistence_snapshotter = config_persistence_snapshotter
        self._config_persistence_restorer = config_persistence_restorer
        self._using_path_persistence_snapshotter = using_path_persistence_snapshotter
        self._using_path_persistence_restorer = using_path_persistence_restorer
        if using_default_config_writer:
            self._config_persistence_snapshotter = (
                self._config_persistence_snapshotter
                or (
                    lambda path: LoadUiConfig._capture_file_bytes(
                        path,
                        coordinator=self._path_transaction_coordinator,
                    )
                )
            )
            self._config_persistence_restorer = (
                self._config_persistence_restorer
                or (
                    lambda path, state: (
                        LoadUiConfig._restore_file_bytes_atomically(
                            path,
                            state,
                            coordinator=self._path_transaction_coordinator,
                        )
                    )
                )
            )
        if using_default_path_writer:
            self._using_path_persistence_snapshotter = (
                self._using_path_persistence_snapshotter
                or (
                    lambda: LoadUiConfig._capture_file_bytes(
                        SEQUENCE_CONFIG_REGISTRY_PATH,
                        coordinator=self._path_transaction_coordinator,
                    )
                )
            )
            self._using_path_persistence_restorer = (
                self._using_path_persistence_restorer
                or (
                    lambda state, expected_current=None: (
                        LoadUiConfig._restore_sequence_registry_checkpoint(
                            SEQUENCE_CONFIG_REGISTRY_PATH,
                            state,
                            expected_current=expected_current,
                            coordinator=self._path_transaction_coordinator,
                        )
                    )
                )
            )
        self._analysis_flag_projection_service = (
            analysis_flag_projection_service
            or SequenceAnalysisFlagProjectionService(
                DataStructAnalysisFlagProjectionPort(model.data_struct)
            )
        )
        self._analysis_config_changed = analysis_config_changed
        self._refresh_test_mode_availability = refresh_test_mode_availability
        self._logger = logger or LogManager.set_log_handler("core")
        self._path_exists = path_exists
        # First-seen ordinals make nested restoration order irrelevant. Cleared
        # reasons remain indexed only while an outstanding checkpoint refers to
        # them; re-adding a fully pruned reason starts a new ordering epoch.
        self._failure_reason_ordinals: dict[tuple[str, str, str], int] = {}
        self._failure_reason_checkpoint_refs: dict[
            tuple[str, str, str], int
        ] = {}
        self._next_failure_reason_ordinal = 0
        self._active_configuration_transaction_token: object | None = None
        self._persistence_failure_reasons: dict[str, tuple[str, ...]] = {}
        self._projection_failure_reasons: dict[str, tuple[str, ...]] = {}
        self._persistence_failures: dict[str, str] = {}
        self._projection_failures: dict[str, str] = {}
        self._persistence_consistent = True
        self._persistence_failure: str | None = None
        self._projection_consistent = True
        self._projection_failure: str | None = None

    @property
    def persistence_consistent(self) -> bool:
        return self._persistence_consistent

    @property
    def persistence_failure(self) -> str | None:
        return self._persistence_failure

    @property
    def projection_consistent(self) -> bool:
        return self._projection_consistent

    @property
    def projection_failure(self) -> str | None:
        return self._projection_failure

    @staticmethod
    def _aggregate_consistency_failures(
        failures: dict[str, str],
        surface_order: tuple[str, ...],
    ) -> str | None:
        ordered_surfaces = [
            surface for surface in surface_order if surface in failures
        ]
        ordered_surfaces.extend(
            sorted(set(failures).difference(ordered_surfaces))
        )
        if not ordered_surfaces:
            return None
        if len(ordered_surfaces) == 1:
            return failures[ordered_surfaces[0]]
        return "; ".join(
            f"{surface}: {failures[surface]}"
            for surface in ordered_surfaces
        )

    def _failure_reason_ordinal(
        self,
        category: str,
        surface: str,
        reason: str,
    ) -> int:
        key = self._failure_reason_key(category, surface, reason)
        ordinal = self._failure_reason_ordinals.get(key)
        if ordinal is None:
            ordinal = self._next_failure_reason_ordinal
            self._failure_reason_ordinals[key] = ordinal
            self._next_failure_reason_ordinal += 1
        return ordinal

    def _ordered_failure_reasons(
        self,
        category: str,
        surface: str,
        reasons: Iterable[str],
    ) -> tuple[str, ...]:
        unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        return tuple(
            sorted(
                unique_reasons,
                key=lambda reason: self._failure_reason_ordinal(
                    category,
                    surface,
                    reason,
                ),
            )
        )

    @staticmethod
    def _failure_reason_key(
        category: str,
        surface: str,
        reason: str,
    ) -> tuple[str, str, str]:
        return str(category), str(surface), str(reason)

    @staticmethod
    def _checkpoint_failure_reason_keys(
        category: str,
        failures: dict[str, tuple[str, ...]],
    ) -> Iterable[tuple[str, str, str]]:
        for surface, reasons in failures.items():
            for reason in reasons:
                yield str(category), str(surface), str(reason)

    def _retain_failure_reason_checkpoints(
        self,
        projection_failures: dict[str, tuple[str, ...]],
        persistence_failures: dict[str, tuple[str, ...]],
    ) -> None:
        checkpoint_groups = (
            ("projection", projection_failures),
            ("persistence", persistence_failures),
        )
        for category, failures in checkpoint_groups:
            for key in self._checkpoint_failure_reason_keys(
                category, failures
            ):
                self._failure_reason_ordinal(*key)
                self._failure_reason_checkpoint_refs[key] = (
                    self._failure_reason_checkpoint_refs.get(key, 0) + 1
                )

    def _release_failure_reason_checkpoints(
        self,
        projection_failures: dict[str, tuple[str, ...]],
        persistence_failures: dict[str, tuple[str, ...]],
    ) -> None:
        checkpoint_groups = (
            ("projection", projection_failures),
            ("persistence", persistence_failures),
        )
        for category, failures in checkpoint_groups:
            for key in self._checkpoint_failure_reason_keys(
                category, failures
            ):
                retained = self._failure_reason_checkpoint_refs.get(key, 0)
                if retained <= 1:
                    self._failure_reason_checkpoint_refs.pop(key, None)
                else:
                    self._failure_reason_checkpoint_refs[key] = retained - 1
        self._prune_failure_reason_ordering()

    def _active_failure_reason_keys(self) -> set[tuple[str, str, str]]:
        active: set[tuple[str, str, str]] = set()
        for category, failures in (
            ("projection", self._projection_failure_reasons),
            ("persistence", self._persistence_failure_reasons),
        ):
            active.update(
                self._checkpoint_failure_reason_keys(category, failures)
            )
        return active

    def _prune_failure_reason_ordering(self) -> None:
        active = self._active_failure_reason_keys()
        retained = self._failure_reason_checkpoint_refs
        for key in tuple(self._failure_reason_ordinals):
            if key not in active and key not in retained:
                self._failure_reason_ordinals.pop(key, None)

    def _sync_consistency_diagnostics(self) -> None:
        self._projection_failure_reasons = {
            surface: self._ordered_failure_reasons(
                "projection", surface, reasons
            )
            for surface, reasons in self._projection_failure_reasons.items()
            if reasons
        }
        self._persistence_failure_reasons = {
            surface: self._ordered_failure_reasons(
                "persistence", surface, reasons
            )
            for surface, reasons in self._persistence_failure_reasons.items()
            if reasons
        }
        self._projection_failures = {
            surface: "; ".join(reasons)
            for surface, reasons in self._projection_failure_reasons.items()
            if reasons
        }
        self._persistence_failures = {
            surface: "; ".join(reasons)
            for surface, reasons in self._persistence_failure_reasons.items()
            if reasons
        }
        self._projection_consistent = not self._projection_failures
        self._projection_failure = self._aggregate_consistency_failures(
            self._projection_failures,
            _PROJECTION_SURFACE_ORDER,
        )
        self._persistence_consistent = not self._persistence_failures
        self._persistence_failure = self._aggregate_consistency_failures(
            self._persistence_failures,
            _PERSISTENCE_SURFACE_ORDER,
        )
        self._prune_failure_reason_ordering()

    def _record_projection_failure(self, surface: str, reason: str) -> None:
        surface = str(surface)
        reasons = self._projection_failure_reasons.get(surface, ())
        reason = str(reason)
        if reason not in reasons:
            self._projection_failure_reasons[surface] = (*reasons, reason)
        self._sync_consistency_diagnostics()

    def _clear_projection_failure(self, surface: str) -> None:
        self._projection_failure_reasons.pop(str(surface), None)
        self._sync_consistency_diagnostics()

    def _mark_persistence_inconsistent(
        self, surface: str, reason: str
    ) -> None:
        surface = str(surface)
        reasons = self._persistence_failure_reasons.get(surface, ())
        reason = str(reason)
        if reason not in reasons:
            self._persistence_failure_reasons[surface] = (*reasons, reason)
        self._sync_consistency_diagnostics()
        self._disable_inconsistent_configuration()

    def _clear_persistence_failure(self, surface: str) -> None:
        self._persistence_failure_reasons.pop(str(surface), None)
        self._sync_consistency_diagnostics()

    def _actions_may_be_enabled(self) -> bool:
        return bool(
            self._projection_consistent
            and self._persistence_consistent
        )

    def _apply_durable_compensation_result(
        self,
        transaction: _ConfigurationTransaction,
        *,
        compensated: bool,
        failure: str,
        fallback_surface: str | None = None,
    ) -> None:
        surfaces = transaction.durable_surfaces
        if not compensated and not surfaces and fallback_surface is not None:
            surfaces = frozenset({fallback_surface})
        failed_surfaces = transaction.durable_compensation_failure_surfaces
        if not compensated and not failed_surfaces:
            failed_surfaces = surfaces
        for surface in surfaces:
            if surface in failed_surfaces:
                self._persistence_failure_reasons[surface] = (str(failure),)
            else:
                self._persistence_failure_reasons.pop(surface, None)
        self._sync_consistency_diagnostics()
        transaction.restore_failure_provenance()

    def _finalize_projection_success(
        self,
        verified_surfaces: Collection[str],
        *,
        registry_persistence_verified: bool | None,
    ) -> bool:
        """Clear only surfaces proven by successful owning projections."""
        surfaces = frozenset(verified_surfaces)
        repaired = bool(surfaces.intersection(self._projection_failures))
        registry_persistence_repaired = bool(
            "registry" in surfaces
            and registry_persistence_verified is True
            and "registry" in self._persistence_failures
        )
        for surface in surfaces:
            self._clear_projection_failure(surface)
        if "registry" in surfaces and registry_persistence_verified is True:
            self._clear_persistence_failure("registry")
        elif "registry" in surfaces and registry_persistence_verified is False:
            self._mark_persistence_inconsistent(
                "registry", "registry persistence remains unverified"
            )
        if not self._projection_consistent or not self._persistence_consistent:
            self._disable_inconsistent_configuration()
        return bool(repaired or registry_persistence_repaired)

    def _disable_inconsistent_configuration(self) -> None:
        try:
            self.view.set_sequence_config_available(False)
        except Exception as exc:
            self._logger.warning(
                f"Failed to disable inconsistent configuration state: {exc}"
            )

    def _mark_projection_inconsistent(self, surface: str, failure: str) -> None:
        self._record_projection_failure(surface, failure)
        self._logger.warning(f"Failed configuration projection integrity: {failure}")
        self._disable_inconsistent_configuration()

    def _validate_projection_hook_pair(
        self,
        surface: str,
        operation: str,
        callback: Callable[..., Any] | None,
        capturer: Callable[[], Any] | None,
        restorer: Callable[[Any], Any] | None,
    ) -> None:
        if callback is None:
            return
        if capturer is None or restorer is None:
            failure = f"incomplete {operation} projection hooks"
            self._mark_projection_inconsistent(surface, failure)
            raise RuntimeError(failure)

    def _capture_optional_projection_state(
        self,
        operation: str,
        callback: Callable[..., Any] | None,
        capturer: Callable[[], Any] | None,
    ) -> Any:
        if callback is None or capturer is None:
            return None
        try:
            return capturer()
        except Exception as exc:
            raise RuntimeError(f"failed to capture {operation}: {exc}") from exc

    @staticmethod
    def _projection_restorer(
        restorer: Callable[[Any], Any] | None, state: Any
    ) -> Callable[[], Any] | None:
        if restorer is None:
            return None
        return lambda: restorer(state)

    def _configuration_path_exists(self, path: Any) -> bool:
        try:
            return bool(self._path_exists(path))
        except Exception as exc:
            self._logger.warning(
                f"Failed to probe sequence config path {path}: {exc}"
            )
            return False

    def _capture_persistence_checkpoint(
        self, operation: str, snapshotter: Callable[..., Any] | None, *args: Any
    ) -> tuple[bool, Any]:
        if snapshotter is None:
            return True, None
        try:
            return True, snapshotter(*args)
        except Exception as exc:
            self._logger.warning(f"Failed to capture {operation}: {exc}")
            return False, None

    def _validated_adapter_override(
        self,
        adapter: Any,
        *,
        operation: str,
    ) -> PersistenceAdapterProtocol | None:
        required_methods = (
            "transaction",
            "capture",
            "write",
            "conditional_restore",
            "ownership_token",
            "read_durable_truth",
            "read_semantic_current",
        )
        try:
            coordinator = getattr(adapter, "coordinator")
            methods_are_callable = all(
                callable(getattr(adapter, name)) for name in required_methods
            )
        except Exception as exc:
            self._logger.warning(
                f"Rejected custom {operation}: protocol inspection failed: {exc}"
            )
            return None
        if not methods_are_callable:
            self._logger.warning(
                f"Rejected custom {operation}: incomplete persistence protocol"
            )
            return None
        if (
            not isinstance(coordinator, PathTransactionCoordinator)
            or coordinator is not self._path_transaction_coordinator
        ):
            self._logger.warning(
                f"Rejected custom {operation}: coordinator is not shared"
            )
            return None
        return adapter

    def _run_persistence_context(
        self,
        transaction: _ConfigurationTransaction,
        adapter: PersistenceAdapterProtocol,
        target: Any,
        *,
        operation: str,
        body: Callable[[], bool],
        persistence_surface: str = "configuration_file",
    ) -> bool:
        journal_count = transaction.durable_write_count

        def mark_uncertain_if_unjournaled() -> None:
            if transaction.durable_write_count == journal_count:
                transaction.mark_durable_uncertain(
                    operation, persistence_surface
                )

        def raise_logging_interruption_after_recovery(
            failures: _RecoveryFailureAggregator,
            *,
            failure: str,
        ) -> None:
            if not failures.has_interruption:
                return
            mark_uncertain_if_unjournaled()
            interruption = failures.primary_error
            if interruption is None:
                return
            self._recover_before_base_exception(
                transaction,
                interruption,
                failure=failure,
            )
            failures.raise_if_interrupted()

        try:
            context = adapter.transaction(target)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                primary_traceback = exc.__traceback__
                transaction.mark_durable_uncertain(
                    operation, persistence_surface
                )
                self._recover_before_base_exception(
                    transaction,
                    exc,
                    failure=(
                        f"{operation} setup interrupted before durable "
                        "ownership capture"
                    ),
                )
                exc.__traceback__ = primary_traceback
                raise
            transaction.mark_durable_uncertain(operation, persistence_surface)
            failures = _RecoveryFailureAggregator()
            failures.capture("Persistence context creation", exc)
            failures.warning(
                self._logger,
                f"Failed to create {operation} context: "
                f"{_safe_exception_description(exc)}",
                operation="Persistence context creation diagnostic logging",
            )
            raise_logging_interruption_after_recovery(
                failures,
                failure=f"{operation} setup diagnostic logging interrupted",
            )
            return False
        enter = getattr(context, "__enter__", None)
        exit_context = getattr(context, "__exit__", None)
        if not callable(enter) or not callable(exit_context):
            transaction.mark_durable_uncertain(operation, persistence_surface)
            failures = _RecoveryFailureAggregator()
            failures.warning(
                self._logger,
                f"Failed to create {operation} context: invalid context manager",
                operation="Invalid persistence context diagnostic logging",
            )
            raise_logging_interruption_after_recovery(
                failures,
                failure=f"{operation} setup diagnostic logging interrupted",
            )
            return False
        try:
            enter()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                primary_traceback = exc.__traceback__
                transaction.mark_durable_uncertain(
                    operation, persistence_surface
                )
                self._recover_before_base_exception(
                    transaction,
                    exc,
                    failure=(
                        f"{operation} setup interrupted before durable "
                        "ownership capture"
                    ),
                )
                exc.__traceback__ = primary_traceback
                raise
            transaction.mark_durable_uncertain(operation, persistence_surface)
            failures = _RecoveryFailureAggregator()
            failures.capture("Persistence context entry", exc)
            failures.warning(
                self._logger,
                f"Failed to enter {operation} context: "
                f"{_safe_exception_description(exc)}",
                operation="Persistence context entry diagnostic logging",
            )
            raise_logging_interruption_after_recovery(
                failures,
                failure=f"{operation} setup diagnostic logging interrupted",
            )
            return False

        def record_exit_failure(
            primary_error: BaseException, exit_error: BaseException
        ) -> _RecoveryFailureAggregator:
            failures = _RecoveryFailureAggregator(
                primary_error,
                operation="Persistence context body",
            )
            failures.capture("Persistence context cleanup", exit_error)
            failures.warning(
                self._logger,
                f"Failed to exit {operation} context while handling "
                f"{type(primary_error).__name__}: "
                f"{_safe_exception_description(exit_error)}",
                operation="Persistence cleanup diagnostic logging",
            )
            return failures

        body_started = False
        body_completed = False
        try:
            body_started = True
            result = body()
            body_completed = True
        except BaseException as body_error:
            body_traceback = body_error.__traceback__
            exit_failed = False
            try:
                exit_context(
                    type(body_error),
                    body_error,
                    body_traceback,
                )
            except BaseException as exit_error:
                exit_failed = True
                if (
                    isinstance(body_error, Exception)
                    and not isinstance(exit_error, Exception)
                ):
                    exit_traceback = exit_error.__traceback__
                    failures = _RecoveryFailureAggregator(
                        exit_error,
                        operation="Persistence context cleanup",
                    )
                    failures.capture("Persistence context body", body_error)
                    failures.warning(
                        self._logger,
                        f"Failed to exit {operation} context while handling "
                        f"{type(body_error).__name__}: "
                        f"{_safe_exception_description(exit_error)}",
                        operation="Persistence cleanup diagnostic logging",
                    )
                    mark_uncertain_if_unjournaled()
                    self._recover_before_base_exception(
                        transaction,
                        exit_error,
                        failure=(
                            f"{operation} cleanup interrupted while handling "
                            "body failure"
                        ),
                    )
                    exit_error.__traceback__ = exit_traceback
                    raise
                exit_failures = record_exit_failure(body_error, exit_error)
                if isinstance(body_error, Exception):
                    raise_logging_interruption_after_recovery(
                        exit_failures,
                        failure=(
                            f"{operation} cleanup logging interrupted while "
                            "handling body failure"
                        ),
                    )
                body_error.__traceback__ = body_traceback

            if not isinstance(body_error, Exception):
                mark_uncertain_if_unjournaled()
                self._recover_before_base_exception(
                    transaction,
                    body_error,
                    failure=f"{operation} interrupted before body completion",
                )
                body_error.__traceback__ = body_traceback
                raise
            mark_uncertain_if_unjournaled()
            if not exit_failed:
                failures = _RecoveryFailureAggregator()
                failures.capture("Persistence context body", body_error)
                failures.warning(
                    self._logger,
                    f"Failed inside {operation} context: "
                    f"{_safe_exception_description(body_error)}",
                    operation="Persistence body diagnostic logging",
                )
                raise_logging_interruption_after_recovery(
                    failures,
                    failure=f"{operation} body diagnostic logging interrupted",
                )
            return False
        if not body_started or not body_completed:
            mark_uncertain_if_unjournaled()
            return False
        try:
            exit_context(None, None, None)
        except BaseException as exit_error:
            if not isinstance(exit_error, Exception):
                exit_traceback = exit_error.__traceback__
                mark_uncertain_if_unjournaled()
                self._recover_before_base_exception(
                    transaction,
                    exit_error,
                    failure=f"{operation} cleanup interrupted after body completion",
                )
                exit_error.__traceback__ = exit_traceback
                raise
            mark_uncertain_if_unjournaled()
            failures = _RecoveryFailureAggregator()
            failures.capture("Persistence context cleanup", exit_error)
            failures.warning(
                self._logger,
                f"Failed to exit {operation} context: "
                f"{_safe_exception_description(exit_error)}",
                operation="Persistence cleanup diagnostic logging",
            )
            raise_logging_interruption_after_recovery(
                failures,
                failure=f"{operation} cleanup diagnostic logging interrupted",
            )
            return False
        return result is not False

    def _abort_before_base_exception(
        self,
        transaction: _ConfigurationTransaction,
        primary_error: BaseException,
    ) -> None:
        failures = _RecoveryFailureAggregator(
            primary_error,
            operation="persistence interruption",
        )
        restored = False
        try:
            restored = transaction.abort()
        except BaseException as cleanup_error:
            failures.capture(
                "Provisional configuration rollback",
                cleanup_error,
            )
        if not restored:
            failures.capture(
                "Provisional configuration rollback",
                RuntimeError("rollback was incomplete"),
            )
            self._mark_projection_inconsistent_before_reraise(
                failures.primary_error or primary_error,
                "provisional configuration rollback was incomplete",
            )
        if failures.primary_error is not primary_error:
            failures.raise_if_selected()

    def _mark_projection_inconsistent_before_reraise(
        self,
        primary_error: BaseException,
        failure: str,
    ) -> None:
        """Disable an untrusted projection without replacing a primary error."""
        failures = _RecoveryFailureAggregator(
            primary_error,
            operation="persistence interruption",
        )
        self._record_projection_failure("view", failure)
        failures.warning(
            self._logger,
            f"Failed configuration projection integrity: {failure}",
            operation="Projection-integrity diagnostic logging",
        )
        try:
            self.view.set_sequence_config_available(False)
        except BaseException as cleanup_error:
            failures.capture(
                "Failed to disable inconsistent configuration",
                cleanup_error,
            )
        if failures.primary_error is not primary_error:
            failures.raise_if_selected()

    def _finalize_reconciliation_failure_without_primary(
        self,
        error: BaseException,
        *,
        failure: str,
        persistence_uncertain: bool = False,
    ) -> None:
        """Mark/disable after contained failure, promoting cleanup interrupts."""
        if persistence_uncertain:
            self._persistence_failure_reasons["registry"] = (str(failure),)
        self._projection_failure_reasons["registry"] = (str(failure),)
        self._sync_consistency_diagnostics()
        failures = _RecoveryFailureAggregator()
        failures.capture("Reconciliation", error)
        failures.warning(
            self._logger,
            failure,
            operation="Reconciliation diagnostic logging",
        )
        try:
            self.view.set_sequence_config_available(False)
        except BaseException as cleanup_error:
            failures.capture("Reconciliation action disable", cleanup_error)
        failures.raise_if_interrupted()

    def _recover_before_base_exception(
        self,
        transaction: _ConfigurationTransaction,
        primary_error: BaseException,
        *,
        failure: str,
    ) -> None:
        transaction.mark_terminal_recovery_started()
        failures = _RecoveryFailureAggregator(
            primary_error,
            operation="persistence interruption",
        )
        compensated = False
        projections_restored = False
        try:
            compensated = transaction.compensate_durable_writes()
        except BaseException as cleanup_error:
            failures.capture(
                "Durable compensation",
                cleanup_error,
            )
        try:
            projections_restored = transaction.abort()
        except BaseException as cleanup_error:
            failures.capture(
                "Configuration rollback",
                cleanup_error,
            )
        self._apply_durable_compensation_result(
            transaction,
            compensated=compensated,
            failure=failure,
            fallback_surface="configuration_file",
        )
        if not compensated:
            for compensation_failure in transaction.durable_compensation_failures:
                failures.capture(
                    "Durable compensation",
                    RuntimeError(compensation_failure),
                )
            failures.capture(
                "Durable compensation",
                RuntimeError(
                    "compensation was incomplete; reconciled to durable state"
                ),
            )
        if not compensated or not projections_restored:

            def record_reconciliation_failure(
                operation: str, cleanup_error: BaseException
            ) -> None:
                failures.capture(
                    f"Durable-state reconciliation {operation}",
                    cleanup_error,
                )

            reconciliation_succeeded = self._reconcile_to_durable_truth(
                transaction,
                abort_transaction=False,
                failure_handler=record_reconciliation_failure,
                finalize_success=False,
            )
            if not reconciliation_succeeded and self._projection_consistent:
                self._record_projection_failure(
                    "registry",
                    "durable-state reconciliation was incomplete",
                )
        if not projections_restored:
            self._record_projection_failure(
                "view",
                "configuration rollback required durable-state reconciliation",
            )
            failures.warning(
                self._logger,
                "Failed configuration projection integrity: "
                "configuration rollback required durable-state reconciliation",
                operation="Projection-integrity diagnostic logging",
            )
        transaction.restore_failure_provenance()
        if failures.primary_error is not primary_error:
            failures.raise_if_selected()

    def _config_adapter(self) -> PersistenceAdapterProtocol | None:
        if self._config_persistence_adapter_override is not None:
            return self._validated_adapter_override(
                self._config_persistence_adapter_override,
                operation="sequence config persistence",
            )
        if (
            not callable(self._config_saver)
            or not callable(self._config_persistence_snapshotter)
            or not callable(self._config_persistence_restorer)
        ):
            self._logger.warning(
                "Rejected custom sequence config persistence: incomplete protocol"
            )
            return None
        return PersistenceAdapter(
            coordinator=self._path_transaction_coordinator,
            transaction_key=lambda target: target,
            checkpoint_reader=lambda target: (
                self._config_persistence_snapshotter(target)
            ),
            writer=lambda payload, target: self._config_saver(payload, target),
            checkpoint_restorer=lambda target, checkpoint: (
                self._config_persistence_restorer(target, checkpoint)
            ),
            durable_truth_reader=lambda target: self._config_loader(target),
            semantic_reader=self._read_config_semantic_current,
        )

    def _read_config_semantic_current(self, target: Any) -> Any:
        validated = self._load_validated_configuration(target)
        return validated[0] if validated is not None else None

    def _using_path_adapter(
        self, *, allow_semantic_checkpoint_fallback: bool = False
    ) -> PersistenceAdapterProtocol | None:
        if self._using_path_persistence_adapter_override is not None:
            return self._validated_adapter_override(
                self._using_path_persistence_adapter_override,
                operation="active-path persistence",
            )
        if not callable(self._using_path_updater):
            self._logger.warning(
                "Rejected custom active-path persistence: incomplete protocol"
            )
            return None
        has_snapshotter = callable(self._using_path_persistence_snapshotter)
        has_restorer = callable(self._using_path_persistence_restorer)
        if not (has_snapshotter and has_restorer) and not (
            allow_semantic_checkpoint_fallback
            and self._using_path_persistence_snapshotter is None
            and self._using_path_persistence_restorer is None
        ):
            self._logger.warning(
                "Rejected custom active-path persistence: incomplete protocol"
            )
            return None

        def capture_registry_checkpoint() -> Any:
            if callable(self._using_path_persistence_snapshotter):
                return self._using_path_persistence_snapshotter()
            return _copy_plain_configuration(self._registry_loader() or {})

        def restore_registry_checkpoint(checkpoint: Any) -> Any:
            if callable(self._using_path_persistence_restorer):
                return self._using_path_persistence_restorer(checkpoint)
            current = _copy_plain_configuration(self._registry_loader() or {})
            return (
                _canonical_json_semantic_token(current)
                == _canonical_json_semantic_token(checkpoint)
            )

        return PersistenceAdapter(
            coordinator=self._path_transaction_coordinator,
            transaction_key=lambda _target: (
                self._using_path_persistence_transaction_key
            ),
            checkpoint_reader=lambda _target: capture_registry_checkpoint(),
            writer=lambda payload, _target: self._using_path_updater(payload),
            checkpoint_restorer=lambda _target, checkpoint: (
                restore_registry_checkpoint(checkpoint)
            ),
            durable_truth_reader=lambda _target: self._registry_loader(),
            semantic_reader=self._read_using_path_semantic_current,
        )

    def _read_using_path_semantic_current(self, _target: Any) -> Any:
        return dict(self._registry_loader() or {})

    @staticmethod
    def _active_path_semantic_projection(
        registry: Any,
        checkpoint: _ActivePathSemanticCheckpoint,
    ) -> list[Any]:
        if type(registry) is not dict or type(checkpoint.selected_key) is not str:
            raise TypeError("active-path semantic state requires a JSON object")
        using_present = "using_config_path" in registry
        selected_present = checkpoint.selected_key in registry
        return [
            using_present,
            registry.get("using_config_path"),
            checkpoint.selected_key,
            selected_present,
            registry.get(checkpoint.selected_key),
        ]

    def _semantic_checkpoint_matches(
        self,
        adapter: PersistenceAdapterProtocol,
        target: Any,
        semantic_checkpoint: Any,
        *,
        operation: str,
    ) -> bool:
        try:
            semantic_current = adapter.read_semantic_current(target)
            if isinstance(semantic_checkpoint, _ActivePathSemanticCheckpoint):
                semantic_current = self._active_path_semantic_projection(
                    semantic_current,
                    semantic_checkpoint,
                )
                semantic_checkpoint = [
                    semantic_checkpoint.using_path_present,
                    semantic_checkpoint.using_config_path,
                    semantic_checkpoint.selected_key,
                    semantic_checkpoint.selected_path_present,
                    semantic_checkpoint.selected_path,
                ]
            current_token = _canonical_json_semantic_token(semantic_current)
            checkpoint_token = _canonical_json_semantic_token(
                semantic_checkpoint
            )
        except Exception as exc:
            self._logger.warning(
                f"Failed to capture canonical semantic state for "
                f"{operation}: {exc}"
            )
            return False
        if current_token != checkpoint_token:
            self._logger.warning(
                f"Rejected {operation}: durable semantic state changed "
                "after admission"
            )
            return False
        return True

    def _attempt_config_persistence(
        self,
        transaction: _ConfigurationTransaction,
        sequence_config: list[Any],
        using_config_path: Any,
        *,
        semantic_checkpoint: list[Any],
    ) -> bool:
        adapter = self._config_adapter()
        if adapter is None:
            return False
        self._last_config_persistence_adapter = adapter
        operation = "sequence config persistence"

        def attempt() -> bool:
            captured, checkpoint = self._capture_persistence_checkpoint(
                "sequence config persistence checkpoint",
                adapter.capture,
                using_config_path,
            )
            if not captured:
                transaction.mark_durable_uncertain(
                    "sequence config persistence checkpoint",
                    "configuration_file",
                )
                return False
            if not self._semantic_checkpoint_matches(
                adapter,
                using_config_path,
                semantic_checkpoint,
                operation=operation,
            ):
                return False

            def rollback(expected_current: Any) -> Any:
                return adapter.conditional_restore(
                    using_config_path,
                    checkpoint,
                    expected_current,
                )

            return transaction.attempt_durable_write(
                "write back regenerated stimulus path to config",
                adapter.write,
                sequence_config,
                using_config_path,
                rollback=rollback,
                state_reader=lambda: adapter.capture(using_config_path),
                ownership_tokenizer=adapter.ownership_token,
                surface="configuration_file",
            )
        succeeded = self._run_persistence_context(
            transaction,
            adapter,
            using_config_path,
            operation=operation,
            body=attempt,
            persistence_surface="configuration_file",
        )
        if succeeded:
            self._clear_persistence_failure("configuration_file")
        return succeeded

    def _attempt_using_path_persistence(
        self,
        transaction: _ConfigurationTransaction,
        path: Any,
        *,
        semantic_checkpoint: Any,
        allow_semantic_checkpoint_fallback: bool = False,
        persistence_adapter: PersistenceAdapterProtocol | None = None,
    ) -> _UsingPathPersistenceResult:
        adapter = persistence_adapter
        if adapter is None:
            adapter = self._using_path_adapter(
                allow_semantic_checkpoint_fallback=(
                    allow_semantic_checkpoint_fallback
                )
            )
        if adapter is None:
            return _UsingPathPersistenceResult(False)
        self._last_using_path_persistence_adapter = adapter
        operation = "active sequence config path persistence"
        committed_registry: dict[Any, Any] | None = None

        def attempt() -> bool:
            nonlocal committed_registry
            captured, checkpoint = self._capture_persistence_checkpoint(
                "active sequence config path persistence checkpoint",
                adapter.capture,
                path,
            )
            if not captured:
                transaction.mark_durable_uncertain(
                    "active sequence config path persistence checkpoint",
                    "registry",
                )
                return False
            if not self._semantic_checkpoint_matches(
                adapter,
                path,
                semantic_checkpoint,
                operation=operation,
            ):
                return False

            def rollback(expected_current: Any) -> Any:
                return adapter.conditional_restore(
                    path,
                    checkpoint,
                    expected_current,
                )

            if not transaction.attempt_durable_write(
                f"persist active sequence config path: {path}",
                adapter.write,
                path,
                path,
                rollback=rollback,
                state_reader=lambda: adapter.capture(path),
                ownership_tokenizer=adapter.ownership_token,
                surface="registry",
            ):
                return False
            try:
                durable_registry = adapter.read_durable_truth(path)
                committed_registry = self._validate_committed_active_path_registry(
                    durable_registry,
                    path=path,
                    semantic_checkpoint=semantic_checkpoint,
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to capture committed active-path registry inside "
                    f"transaction: {exc}"
                )
                committed_registry = None
            if committed_registry is None:
                self._logger.warning(
                    "Rejected active-path persistence: post-write durable "
                    "registry did not match the admitted binding"
                )
                return False
            return True

        succeeded = self._run_persistence_context(
            transaction,
            adapter,
            path,
            operation=operation,
            body=attempt,
            persistence_surface="registry",
        )
        return _UsingPathPersistenceResult(
            succeeded,
            committed_registry if succeeded else None,
        )

    def _validate_committed_active_path_registry(
        self,
        durable_registry: Any,
        *,
        path: Any,
        semantic_checkpoint: Any,
    ) -> dict[Any, Any] | None:
        if (
            type(durable_registry) is not dict
            or not isinstance(
                semantic_checkpoint, _ActivePathSemanticCheckpoint
            )
            or not semantic_checkpoint.selected_path_present
        ):
            return None
        try:
            registry = _copy_plain_configuration(durable_registry)
            current_projection = self._active_path_semantic_projection(
                registry,
                semantic_checkpoint,
            )
            expected_projection = [
                True,
                path,
                semantic_checkpoint.selected_key,
                True,
                semantic_checkpoint.selected_path,
            ]
            current_token = _canonical_json_semantic_token(
                current_projection
            )
            expected_token = _canonical_json_semantic_token(
                expected_projection
            )
        except (TypeError, ValueError):
            return None
        return registry if current_token == expected_token else None

    def _project_committed_active_path_registry(
        self,
        transaction: _ConfigurationTransaction,
        persistence_result: _UsingPathPersistenceResult,
        *,
        using_config_path: Any,
    ) -> bool:
        failures = _RecoveryFailureAggregator()
        registry = persistence_result.committed_registry
        if registry is None:
            error = RuntimeError(
                "Cannot project active-path persistence without locked "
                "durable registry truth"
            )
            self._record_aggregated_failure(
                failures,
                "committed registry truth",
                error,
            )
        else:
            self._attempt_registry_projection(
                transaction,
                registry,
                using_config_path=using_config_path,
                failures=failures,
                operation="committed active-path registry",
            )
        if not failures.has_failures:
            return True
        return self._recover_registry_projection_failure(
            transaction,
            failures,
            failure="project committed active sequence config registry failed",
        )

    def _record_aggregated_failure(
        self,
        failures: _RecoveryFailureAggregator,
        operation: str,
        error: BaseException,
    ) -> None:
        failures.capture(operation, error)
        failures.warning(
            self._logger,
            f"Failed to {operation}: {_safe_exception_description(error)}",
            operation=f"{operation} diagnostic logging",
        )

    def _derive_ordered_registry_entries(
        self,
        registry: dict[Any, Any],
        failures: _RecoveryFailureAggregator,
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        """Derive a projection while reporting every probe failure to caller.

        No diagnostic failure escapes this boundary. The caller owns recovery
        and decides whether the selected BaseException must be re-raised.
        """
        starting_failure_count = failures.failure_count
        try:
            keys = [key for key in registry if key != "using_config_path"]
            ordered_keys: list[Any] = []
            if "默认配置" in keys:
                ordered_keys.append("默认配置")
                keys.remove("默认配置")
            ordered_keys.extend(sorted(keys))
        except BaseException as error:
            self._record_aggregated_failure(
                failures,
                "derive ordered sequence config registry keys",
                error,
            )
            return (), False

        visible: list[tuple[str, str]] = []
        for key in ordered_keys:
            try:
                value = registry.get(key)
                if not isinstance(value, str):
                    continue
                exists = True
                if value:
                    exists = bool(self._path_exists(value))
            except BaseException as error:
                self._record_aggregated_failure(
                    failures,
                    f"probe sequence config registry path {key}",
                    error,
                )
                continue
            if exists:
                visible.append((str(key), value))
        return (
            tuple(visible),
            failures.failure_count == starting_failure_count,
        )

    def _attempt_registry_projection(
        self,
        transaction: _ConfigurationTransaction,
        registry: dict[Any, Any],
        *,
        using_config_path: Any,
        failures: _RecoveryFailureAggregator,
        operation: str,
    ) -> bool:
        try:
            entries, entries_derived = self._derive_ordered_registry_entries(
                registry, failures
            )
        except BaseException as error:
            self._record_aggregated_failure(
                failures,
                f"{operation} entry derivation",
                error,
            )
            return False
        if not entries_derived:
            return False

        model_projected = False
        try:
            self.model.replace_registry(
                registry,
                using_config_path=using_config_path,
                entries=entries,
            )
            model_projected = True
        except BaseException as error:
            self._record_aggregated_failure(
                failures,
                f"{operation} model projection",
                error,
            )

        view_projected = False
        try:
            self.view.populate_configuration_entries(
                entries,
                using_config_path=using_config_path,
                clear_first=True,
            )
            view_projected = True
        except BaseException as error:
            self._record_aggregated_failure(
                failures,
                f"{operation} view projection",
                error,
            )
        projected = bool(model_projected and view_projected)
        if projected:
            transaction.mark_surface_verified("registry")
        return projected

    def _recover_registry_projection_failure(
        self,
        transaction: _ConfigurationTransaction,
        failures: _RecoveryFailureAggregator,
        *,
        failure: str,
    ) -> bool:
        """Recover every boundary after a committed registry projection fails."""
        transaction.mark_terminal_recovery_started()
        compensated = False
        try:
            compensated = transaction.compensate_durable_writes()
        except BaseException as error:
            self._record_aggregated_failure(
                failures, "committed registry durable compensation", error
            )

        restored = False
        try:
            restored = transaction.abort()
            if not restored:
                self._record_aggregated_failure(
                    failures,
                    "committed registry projection rollback",
                    RuntimeError("configuration rollback was incomplete"),
                )
        except BaseException as error:
            self._record_aggregated_failure(
                failures, "committed registry projection rollback", error
            )

        self._apply_durable_compensation_result(
            transaction,
            compensated=compensated,
            failure=failure,
            fallback_surface="registry",
        )

        def capture_reconciliation_failure(
            operation: str, error: BaseException
        ) -> None:
            failures.capture(
                f"committed registry durable-state reconciliation {operation}",
                error,
            )

        reconciled = self._reconcile_to_durable_truth(
            transaction,
            abort_transaction=False,
            failure_handler=capture_reconciliation_failure,
            finalize_success=False,
        )
        if not reconciled:
            self._record_projection_failure(
                "registry",
                "committed registry projection recovery was unverifiable",
            )
        elif restored:
            self._finalize_projection_success(
                {"registry"},
                registry_persistence_verified=(
                    True if compensated else None
                ),
            )

        try:
            self.view.set_sequence_config_available(False)
        except BaseException as error:
            self._record_aggregated_failure(
                failures,
                "disable actions after committed registry projection failure",
                error,
            )
            self._record_projection_failure(
                "view",
                "failed to disable actions after committed registry projection "
                "failure",
            )

        transaction.restore_failure_provenance()
        failures.raise_if_interrupted()
        return False

    def _abort_after_durable_failure(
        self,
        transaction: _ConfigurationTransaction,
        failure: str,
        *,
        persistence_surface: str = "configuration_file",
    ) -> bool:
        transaction.mark_terminal_recovery_started()
        failures = _RecoveryFailureAggregator()
        fatal_ordinary_failure = False

        compensated = False
        try:
            compensated = transaction.compensate_durable_writes()
        except BaseException as exc:
            fatal_ordinary_failure = isinstance(exc, Exception)
            failures.capture("durable compensation", exc)

        projections_restored = False
        try:
            projections_restored = transaction.abort()
        except BaseException as exc:
            fatal_ordinary_failure = (
                fatal_ordinary_failure or isinstance(exc, Exception)
            )
            failures.capture("configuration rollback", exc)

        self._apply_durable_compensation_result(
            transaction,
            compensated=compensated,
            failure=failure,
            fallback_surface=persistence_surface,
        )
        if not compensated or not projections_restored:

            def record_reconciliation_failure(
                operation: str, error: BaseException
            ) -> None:
                failures.capture(
                    f"durable-state reconciliation {operation}",
                    error,
                )

            reconciliation_succeeded = self._reconcile_to_durable_truth(
                transaction,
                abort_transaction=False,
                failure_handler=record_reconciliation_failure,
                finalize_success=False,
            )
            if not reconciliation_succeeded and self._projection_consistent:
                self._record_projection_failure(
                    "registry",
                    "durable-state reconciliation was incomplete",
                )
        if not projections_restored:
            self._record_projection_failure(
                "view",
                "durable failure rollback required reconciliation",
            )
            failures.warning(
                self._logger,
                "Failed configuration projection integrity: "
                "durable failure rollback required reconciliation",
                operation="configuration projection diagnostic",
            )

        transaction.restore_failure_provenance()
        if failures.has_interruption or fatal_ordinary_failure:
            failures.raise_if_selected()
        return False

    def _reconcile_to_durable_truth(
        self,
        transaction: _ConfigurationTransaction,
        *,
        abort_transaction: bool = True,
        failure_handler: Callable[[str, BaseException], Any] | None = None,
        finalize_success: bool = True,
    ) -> bool:
        """Best-effort projection of verified durable truth into a disabled UI."""
        failures: list[tuple[str, BaseException]] = []
        aggregated_failures = _RecoveryFailureAggregator()

        def deliver_failure(operation: str, error: BaseException) -> None:
            failures.append((operation, error))
            aggregated_failures.capture(operation, error)
            if failure_handler is not None:
                try:
                    failure_handler(operation, error)
                except BaseException as handler_error:
                    failures.append(
                        (f"{operation} failure reporting", handler_error)
                    )
                    aggregated_failures.capture(
                        f"{operation} failure reporting",
                        handler_error,
                    )

        def record_failure(operation: str, error: BaseException) -> None:
            deliver_failure(operation, error)
            try:
                self._logger.warning(
                    f"Failed durable-state reconciliation {operation}: "
                    f"{_safe_exception_description(error)}"
                )
            except BaseException as logging_error:
                deliver_failure(
                    f"{operation} diagnostic logging",
                    logging_error,
                )

        abort_succeeded = True
        if abort_transaction:
            try:
                abort_succeeded = transaction.abort()
                if not abort_succeeded:
                    record_failure(
                        "configuration rollback",
                        RuntimeError("configuration rollback was incomplete"),
                    )
            except BaseException as exc:
                abort_succeeded = False
                record_failure("configuration rollback", exc)

        registry_verified = False
        registry: dict[Any, Any] | None = None
        try:
            if self._last_using_path_persistence_adapter is not None:
                durable_registry = (
                    self._last_using_path_persistence_adapter.read_durable_truth(
                        None
                    )
                )
            else:
                durable_registry = self._registry_loader()
            if durable_registry is None:
                registry = {}
            elif type(durable_registry) is dict:
                registry = _copy_plain_configuration(durable_registry)
            else:
                raise TypeError(
                    "durable sequence config registry is not an exact mapping"
                )
            registry_verified = True
        except BaseException as exc:
            record_failure("registry read", exc)

        using_path = (
            registry.get("using_config_path")
            if registry is not None
            else None
        )
        config_verified = False
        validated: tuple[list[Any], dict[Any, Any]] | None = None
        if registry_verified and using_path is None:
            config_verified = True
        elif registry_verified:
            try:
                if self._last_config_persistence_adapter is not None:
                    load_code, result = (
                        self._last_config_persistence_adapter.read_durable_truth(
                            using_path
                        )
                    )
                else:
                    load_code, result = self._config_loader(using_path)
                validated = (
                    self._validated_configuration(result)
                    if load_code == self._ok_code and result
                    else None
                )
                config_verified = True
            except BaseException as exc:
                record_failure("configuration read", exc)

        projection_truth_verified = registry_verified and config_verified
        entries: tuple[tuple[str, Any], ...] = ()
        entries_derived = False
        if projection_truth_verified and registry is not None:
            before_derivation = aggregated_failures.failure_count
            try:
                entries, entries_derived = self._derive_ordered_registry_entries(
                    registry, aggregated_failures
                )
            except BaseException as exc:
                record_failure("registry entry derivation", exc)
                before_derivation = aggregated_failures.failure_count
            if aggregated_failures.failure_count != before_derivation:
                for record in aggregated_failures.records[before_derivation:]:
                    failures.append((record.operation, record.error))
                    if failure_handler is not None:
                        try:
                            failure_handler(record.operation, record.error)
                        except BaseException as handler_error:
                            failures.append(
                                (
                                    f"{record.operation} failure reporting",
                                    handler_error,
                                )
                            )
                            aggregated_failures.capture(
                                f"{record.operation} failure reporting",
                                handler_error,
                            )

        model_projected = False
        projection_attempted = False
        if projection_truth_verified and registry is not None and entries_derived:
            projection_attempted = True
            sequence_config, analysis_config = validated or ([], {})
            try:
                if validated is not None:
                    runtime_candidate = _copy_plain_configuration(
                        sequence_config
                    )
                    prepared, _regenerated = self._prepare_stimulus_config(
                        runtime_candidate, using_path
                    )
                    if not prepared:
                        self._clear_stimulus_runtime_state(
                            clear_sample_rate=True
                        )
                self.model.replace_registry(
                    registry,
                    using_config_path=using_path,
                    entries=entries,
                )
                snapshot = self._build_configuration_snapshot(
                    sequence_config,
                    analysis_config,
                    using_config_path=using_path,
                )
                if not self.model.apply_configuration(
                    snapshot,
                    generation=(
                        transaction.checkpoint.configuration_generation + 1
                    ),
                ):
                    raise RuntimeError(
                        "durable configuration model projection was rejected"
                    )
                model_projected = True
            except BaseException as exc:
                record_failure("model projection", exc)

        view_projected = False
        if projection_truth_verified and entries_derived:
            try:
                self.view.populate_configuration_entries(
                    entries,
                    using_config_path=using_path,
                    clear_first=True,
                )
                view_projected = True
            except BaseException as exc:
                record_failure("view projection", exc)

        projection_operation_failed = bool(failures)
        configuration_dependencies_changed = False
        if projection_truth_verified:
            sequence_config, analysis_config = validated or ([], {})
            try:
                durable_projection_token = _canonical_json_semantic_token(
                    {
                        "using_config_path": using_path,
                        "sequence_config": sequence_config,
                        "analysis_config": analysis_config,
                    }
                )
                checkpoint_projection_token = _canonical_json_semantic_token(
                    {
                        "using_config_path": (
                            transaction.checkpoint.using_config_path
                        ),
                        "sequence_config": (
                            transaction.checkpoint.sequence_config
                        ),
                        "analysis_config": (
                            transaction.checkpoint.analysis_config
                        ),
                    }
                )
                configuration_dependencies_changed = (
                    durable_projection_token != checkpoint_projection_token
                )
            except (TypeError, ValueError):
                configuration_dependencies_changed = True
            if configuration_dependencies_changed:
                record_failure(
                    "configuration-dependent projections",
                    RuntimeError(
                        "durable configuration changed without verified channel, "
                        "analysis, mode, runtime, and presentation reprojection"
                    ),
                )

        if projection_operation_failed and projection_attempted:
            try:
                restored_after_projection_failure = transaction.abort()
                if not restored_after_projection_failure:
                    record_failure(
                        "projection rollback",
                        RuntimeError(
                            "configuration rollback after reconciliation "
                            "projection failure was incomplete"
                        ),
                    )
            except BaseException as exc:
                record_failure("projection rollback", exc)

        disable_succeeded = False
        try:
            self.view.set_sequence_config_available(False)
            disable_succeeded = True
        except BaseException as exc:
            record_failure("action disable", exc)

        succeeded = bool(
            abort_succeeded
            and projection_truth_verified
            and model_projected
            and view_projected
            and disable_succeeded
            and not failures
        )
        if not succeeded:
            failed_operations = ", ".join(
                dict.fromkeys(operation for operation, _error in failures)
            )
            failure = (
                "durable-state reconciliation was incomplete"
                + (f": {failed_operations}" if failed_operations else "")
            )
            surface = (
                "configuration"
                if configuration_dependencies_changed
                else "registry"
            )
            self._record_projection_failure(surface, failure)
        elif finalize_success:
            self._finalize_projection_success(
                {"registry"},
                registry_persistence_verified=True,
            )
            transaction.commit_failure_provenance()
        else:
            transaction.restore_failure_provenance()
        if not succeeded:
            transaction.restore_failure_provenance()
        if failure_handler is None:
            aggregated_failures.raise_if_interrupted()
        return succeeded

    def _finish_configuration_transaction(
        self,
        transaction: _ConfigurationTransaction,
        event: ConfigurationChanged,
        *,
        registry_persistence_verified: bool | None = None,
    ) -> bool:
        failure: str | None = None
        try:
            if not transaction.run_commit_callbacks():
                failure = "required configuration cleanup failed"
            if failure is None:
                repaired = self._finalize_projection_success(
                    transaction.verified_surfaces,
                    registry_persistence_verified=registry_persistence_verified,
                )
                if repaired and not transaction.attempt(
                    "present repaired configuration availability",
                    self.present_configuration_availability,
                    bool(self.model.sequence_config),
                    rollback=lambda: self.view.restore_action_availability(
                        transaction.view_state.action_availability
                    ),
                    required_projection=True,
                    surface="view",
                ):
                    failure = "required configuration presentation failed"
                elif not repaired and not self._actions_may_be_enabled():
                    self._disable_inconsistent_configuration()
        except BaseException as error:
            transaction.recover_base_exception(
                error,
                failure="required configuration commit was interrupted",
            )
            raise
        if failure is not None:
            return self._abort_after_durable_failure(
                transaction, failure
            )
        transaction.commit_failure_provenance()
        self._publish_applied_configuration(event)
        return True

    def _finish_verified_registry_projection_transaction(
        self,
        transaction: _ConfigurationTransaction,
        event: ConfigurationChanged,
    ) -> bool:
        """Commit only after the active path and every projection are verified."""
        failure: str | None = None
        try:
            if not transaction.run_commit_callbacks():
                failure = "required configuration cleanup failed"
            if failure is None:
                self._finalize_projection_success(
                    transaction.verified_surfaces,
                    registry_persistence_verified=True,
                )
                if not transaction.attempt(
                    "present verified configuration availability",
                    self.present_configuration_availability,
                    bool(self.model.sequence_config),
                    rollback=lambda: self.view.restore_action_availability(
                        transaction.view_state.action_availability
                    ),
                    required_projection=True,
                    surface="view",
                ):
                    failure = "required configuration presentation failed"
        except BaseException as error:
            transaction.recover_base_exception(
                error,
                failure="required configuration commit was interrupted",
            )
            raise
        if failure is not None:
            return self._abort_after_durable_failure(
                transaction, failure
            )
        transaction.commit_failure_provenance()
        self._publish_applied_configuration(event)
        return True

    def _ordered_registry_entries(
        self, registry: dict[Any, Any]
    ) -> tuple[tuple[str, str], ...]:
        failures = _RecoveryFailureAggregator()
        entries, _succeeded = self._derive_ordered_registry_entries(
            registry, failures
        )
        failures.raise_if_interrupted()
        return entries

    def _load_registry_candidate(
        self,
    ) -> tuple[
        Any,
        dict[Any, Any],
        tuple[tuple[str, str], ...],
        bool,
        Any,
    ]:
        try:
            loaded = self._registry_loader()
        except Exception as exc:
            self._logger.warning(f"Failed to load sequence config registry: {exc}")
            loaded = {}
        registry = dict(loaded or {})
        user_keys = [
            key
            for key in registry
            if key not in ("using_config_path", "默认配置")
        ]
        using_config_path = registry.get("using_config_path")
        default_path = registry.get("默认配置")
        persistence_required = False
        selected_key = None
        if not using_config_path or (
            isinstance(using_config_path, str)
            and not self._configuration_path_exists(using_config_path)
        ):
            fallback_path = None
            for key in sorted(user_keys):
                path = registry.get(key)
                if isinstance(path, str) and self._configuration_path_exists(path):
                    fallback_path = path
                    selected_key = key
                    break
            if (
                fallback_path is None
                and isinstance(default_path, str)
                and self._configuration_path_exists(default_path)
            ):
                fallback_path = default_path
                selected_key = "默认配置"
            using_config_path = fallback_path
            persistence_required = bool(using_config_path)
        entries = self._ordered_registry_entries(registry)
        return (
            using_config_path,
            registry,
            entries,
            persistence_required,
            selected_key,
        )

    def get_sequence_config_from_registry(
        self,
    ) -> tuple[Any, dict[Any, Any]] | None:
        if self._reject_reentrant_configuration_entry(
            "sequence configuration registry load"
        ):
            return None
        previous_path = self.model.using_config_path
        (
            using_config_path,
            registry,
            entries,
            persistence_required,
            selected_key,
        ) = self._load_registry_candidate()
        if persistence_required:
            persistence_adapter = self._using_path_adapter(
                allow_semantic_checkpoint_fallback=True
            )
            if persistence_adapter is None:
                self.view.restore_selection(previous_path)
                return previous_path, _copy_plain_configuration(registry)
            transaction = self._begin_configuration_transaction(
                selection_path=previous_path
            )
            if transaction is None:
                self.view.restore_selection(previous_path)
                return previous_path, _copy_plain_configuration(registry)
            result = self._run_owned_configuration_transaction(
                transaction,
                "startup configuration fallback",
                self._get_sequence_config_from_registry_transaction,
                previous_path,
                using_config_path,
                registry,
                selected_key,
                persistence_adapter,
            )
            if result is _RECONCILE_REGISTRY_AFTER_TRANSACTION:
                return self._reconcile_registry_projection_to_durable_truth()
            return result
        else:
            self.model.replace_registry(
                registry,
                using_config_path=using_config_path,
                entries=entries,
            )
        return using_config_path, _copy_plain_configuration(registry)

    def _get_sequence_config_from_registry_transaction(
        self,
        transaction: _ConfigurationTransaction,
        previous_path: Any,
        using_config_path: Any,
        registry: dict[Any, Any],
        selected_key: Any,
        persistence_adapter: PersistenceAdapterProtocol,
    ) -> tuple[Any, dict[Any, Any]] | _ReconcileRegistryAfterTransaction:
        semantic_checkpoint = _ActivePathSemanticCheckpoint(
            using_path_present="using_config_path" in registry,
            using_config_path=registry.get("using_config_path"),
            selected_key=selected_key,
            selected_path_present=selected_key in registry,
            selected_path=registry.get(selected_key),
        )
        persistence_result = self._attempt_using_path_persistence(
            transaction,
            using_config_path,
            semantic_checkpoint=semantic_checkpoint,
            allow_semantic_checkpoint_fallback=True,
            persistence_adapter=persistence_adapter,
        )
        if not persistence_result:
            if transaction.durable_write_count == 0:
                transaction.abort()
                return _RECONCILE_REGISTRY_AFTER_TRANSACTION
            self._abort_after_durable_failure(
                transaction,
                "startup active sequence config path persistence failed",
                persistence_surface="registry",
            )
            if not self.persistence_consistent:
                return (
                    self.model.using_config_path,
                    _copy_plain_configuration(self.model.registry),
                )
            return previous_path, _copy_plain_configuration(registry)
        committed_registry = persistence_result.committed_registry
        if committed_registry is None:
            failures = _RecoveryFailureAggregator()
            self._record_aggregated_failure(
                failures,
                "startup committed registry truth",
                RuntimeError(
                    "successful active-path persistence omitted committed registry"
                ),
            )
            self._recover_registry_projection_failure(
                transaction,
                failures,
                failure="startup active sequence config registry projection failed",
            )
            return (
                self.model.using_config_path,
                _copy_plain_configuration(self.model.registry),
            )
        if not self._project_committed_active_path_registry(
            transaction,
            persistence_result,
            using_config_path=using_config_path,
        ):
            return (
                self.model.using_config_path,
                _copy_plain_configuration(self.model.registry),
            )
        self._finalize_projection_success(
            transaction.verified_surfaces,
            registry_persistence_verified=True,
        )
        transaction.commit_failure_provenance()
        return using_config_path, _copy_plain_configuration(committed_registry)

    def _reconcile_registry_projection_to_durable_truth(
        self,
    ) -> tuple[Any, dict[Any, Any]]:
        if self._reject_reentrant_configuration_entry(
            "sequence configuration registry reconciliation"
        ):
            return (
                self.model.using_config_path,
                _copy_plain_configuration(self.model.registry),
            )
        adapter = self._last_using_path_persistence_adapter
        retained_path = self.model.using_config_path
        retained_registry = _copy_plain_configuration(self.model.registry)
        try:
            durable_registry = (
                adapter.read_durable_truth(None)
                if adapter is not None
                else self._registry_loader()
            )
            if durable_registry is None:
                registry = {}
            elif type(durable_registry) is dict:
                registry = _copy_plain_configuration(durable_registry)
            else:
                raise TypeError(
                    "durable sequence config registry is not an exact mapping"
                )
        except BaseException as read_error:
            failure = (
                "startup active sequence config path durable truth is unavailable"
            )
            self._finalize_reconciliation_failure_without_primary(
                read_error,
                failure=failure,
                persistence_uncertain=True,
            )
            return retained_path, retained_registry
        transaction = self._begin_configuration_transaction(
            selection_path=retained_path
        )
        if transaction is None:
            return retained_path, retained_registry
        return self._run_owned_configuration_transaction(
            transaction,
            "startup registry reconciliation",
            self._reconcile_registry_projection_transaction,
            registry,
            retained_path,
            retained_registry,
        )

    def _reconcile_registry_projection_transaction(
        self,
        transaction: _ConfigurationTransaction,
        registry: dict[Any, Any],
        retained_path: Any,
        retained_registry: dict[Any, Any],
    ) -> tuple[Any, dict[Any, Any]]:
        using_config_path = registry.get("using_config_path")
        failures = _RecoveryFailureAggregator()
        projected = self._attempt_registry_projection(
            transaction,
            registry,
            using_config_path=using_config_path,
            failures=failures,
            operation="startup durable registry reconciliation",
        )
        if projected and not failures.has_failures:
            self._finalize_projection_success(
                transaction.verified_surfaces,
                registry_persistence_verified=True,
            )
            transaction.commit_failure_provenance()
            return using_config_path, _copy_plain_configuration(registry)
        try:
            restored = transaction.abort()
            if not restored:
                self._record_aggregated_failure(
                    failures,
                    "startup durable registry projection rollback",
                    RuntimeError("configuration rollback was incomplete"),
                )
        except BaseException as rollback_error:
            self._record_aggregated_failure(
                failures,
                "startup durable registry projection rollback",
                rollback_error,
            )
        failure = "failed to reconcile verified sequence config registry projection"
        self._record_projection_failure("registry", failure)
        try:
            self.view.set_sequence_config_available(False)
        except BaseException as disable_error:
            self._record_aggregated_failure(
                failures,
                "startup durable registry reconciliation action disable",
                disable_error,
            )
        if failures.has_interruption:
            transaction.mark_terminal_recovery_started()
        failures.raise_if_interrupted()
        return retained_path, retained_registry

    def update_using_file_combobox(self) -> None:
        if self._reject_reentrant_configuration_entry(
            "sequence configuration selection refresh"
        ):
            return None
        self.get_sequence_config_from_registry()
        self.view.populate_configuration_entries(
            self.model.registry_entries,
            using_config_path=self.model.using_config_path,
            clear_first=True,
        )
        return None

    def add_file_to_using_file_combobox(self) -> None:
        self.view.populate_configuration_entries(
            self.model.registry_entries,
            using_config_path=self.model.using_config_path,
            clear_first=True,
        )

    @staticmethod
    def _validated_configuration(result: Any) -> tuple[list[Any], dict[Any, Any]] | None:
        if not isinstance(result, list) or not result:
            return None
        try:
            sequence = result[0]["seq1"]
            acquisition = sequence["acq"]
            mode = acquisition["mode"]
            detail = acquisition["detail"]
        except (IndexError, KeyError, TypeError):
            return None
        if not isinstance(sequence, dict) or not isinstance(acquisition, dict):
            return None
        if not isinstance(mode, str) or not isinstance(detail, dict):
            return None
        analysis = sequence.get("analysis_list", {})
        if not isinstance(analysis, dict):
            return None
        try:
            copied_result = _copy_plain_configuration(result)
        except TypeError:
            return None
        copied_analysis = copied_result[0]["seq1"].get("analysis_list", {})
        return copied_result, copied_analysis

    def _build_configuration_snapshot(
        self,
        sequence_config: list[Any],
        analysis_config: dict[Any, Any],
        *,
        using_config_path: Any = _CURRENT_CONFIGURATION_PATH,
    ) -> ConfigurationSnapshot:
        candidate_path = (
            self.model.using_config_path
            if using_config_path is _CURRENT_CONFIGURATION_PATH
            else using_config_path
        )
        return ConfigurationSnapshot(
            sequence_config=sequence_config,
            analysis_config=analysis_config,
            mic=self.model.mic,
            speaker=self.model.speaker,
            mic_channels=tuple(self.model.mic_channels),
            using_config_path=candidate_path,
            streaming_stimulus_data=self.model.streaming_stimulus_data,
        )

    def _build_configuration_changed(
        self,
        sequence_config: list[Any],
        analysis_config: dict[Any, Any],
        *,
        using_config_path: Any = _CURRENT_CONFIGURATION_PATH,
    ) -> ConfigurationChanged:
        return ConfigurationChanged(
            configuration_generation=self.model.configuration_generation + 1,
            configuration_snapshot=self._build_configuration_snapshot(
                sequence_config,
                analysis_config,
                using_config_path=using_config_path,
            ),
        )

    def _apply_configuration_candidate(
        self,
        sequence_config: list[Any],
        analysis_config: dict[Any, Any],
        *,
        using_config_path: Any = _CURRENT_CONFIGURATION_PATH,
    ) -> ConfigurationChanged | None:
        event = self._build_configuration_changed(
            sequence_config,
            analysis_config,
            using_config_path=using_config_path,
        )
        if not self.model.apply_configuration(
            event.configuration_snapshot,
            generation=event.configuration_generation,
        ):
            return None
        return event

    def _publish_applied_configuration(self, event: ConfigurationChanged) -> bool:
        if self._publish_configuration is not None:
            try:
                published = self._publish_configuration(event)
            except Exception as exc:
                self._logger.warning(
                    f"Failed to publish sequence configuration: {exc}"
                )
                return False
            if published is False:
                self._logger.warning(
                    "Failed to publish sequence configuration: publisher rejected event"
                )
                return False
        return True

    def _apply_and_publish(
        self,
        sequence_config: list[Any],
        analysis_config: dict[Any, Any],
        *,
        using_config_path: Any = _CURRENT_CONFIGURATION_PATH,
    ) -> bool:
        checkpoint = self.model.checkpoint_configuration_state()
        event = self._apply_configuration_candidate(
            sequence_config,
            analysis_config,
            using_config_path=using_config_path,
        )
        if event is None:
            self.model.restore_configuration_state(checkpoint)
            return False
        self._publish_applied_configuration(event)
        return True

    def _load_validated_configuration(
        self, path: Any
    ) -> tuple[list[Any], dict[Any, Any]] | None:
        try:
            load_code, result = self._config_loader(path)
        except Exception as exc:
            self._logger.warning(f"Failed to load sequence config: {exc}")
            load_code, result = None, None
        return (
            self._validated_configuration(result)
            if load_code == self._ok_code and result
            else None
        )

    def _begin_configuration_transaction(
        self, *, selection_path: Any
    ) -> _ConfigurationTransaction | None:
        if self._reject_reentrant_configuration_entry(
            "configuration transaction"
        ):
            return None
        owner_token = object()
        self._active_configuration_transaction_token = owner_token
        try:
            return _ConfigurationTransaction(
                self,
                selection_path=selection_path,
                owner_token=owner_token,
            )
        except Exception as exc:
            self._release_configuration_transaction_owner(owner_token)
            self._logger.warning(
                f"Failed to capture configuration transaction state: {exc}"
            )
            return None
        except BaseException:
            self._release_configuration_transaction_owner(owner_token)
            raise

    def _reject_reentrant_configuration_entry(self, _operation: str) -> bool:
        return self._active_configuration_transaction_token is not None

    def _release_configuration_transaction_owner(
        self,
        owner_token: object,
    ) -> None:
        if self._active_configuration_transaction_token is owner_token:
            self._active_configuration_transaction_token = None

    def _run_owned_configuration_transaction(
        self,
        transaction: _ConfigurationTransaction,
        operation: str,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        transaction.enter_owner_scope()
        try:
            return callback(transaction, *args, **kwargs)
        except BaseException as error:
            transaction.recover_base_exception(
                error,
                failure=f"{operation} was interrupted",
            )
            raise
        finally:
            transaction.finalize_owner()

    def get_sequence_config_from_json(self, *, present: bool = True) -> None:
        if self._reject_reentrant_configuration_entry(
            "active configuration load"
        ):
            return None
        transaction = self._begin_configuration_transaction(
            selection_path=self.model.using_config_path
        )
        if transaction is None:
            return None
        return self._run_owned_configuration_transaction(
            transaction,
            "active configuration load",
            self._get_sequence_config_from_json_transaction,
            present=present,
        )

    def _get_sequence_config_from_json_transaction(
        self,
        transaction: _ConfigurationTransaction,
        *,
        present: bool,
    ) -> None:

        def fail() -> None:
            transaction.abort()
            return None

        validated = transaction.call_owned(
            "load active sequence configuration",
            self._load_validated_configuration,
            self.model.using_config_path,
        )
        configuration_available = validated is not None
        sequence_config, analysis_config = validated or ([], {})
        persisted_config_checkpoint = _copy_plain_configuration(sequence_config)
        candidate_snapshot = self._build_configuration_snapshot(
            sequence_config,
            analysis_config,
        )
        if not transaction.call_owned(
            "project active sequence configuration candidate",
            self.model.apply_configuration,
            candidate_snapshot,
            generation=transaction.checkpoint.configuration_generation,
        ):
            return fail()

        regenerated = False
        if configuration_available:
            runtime_ready, regenerated = transaction.call_owned(
                "prepare active configuration runtime",
                self._prepare_stimulus_config,
                sequence_config,
                self.model.using_config_path,
            )
            if not runtime_ready:
                return fail()
            effective_snapshot = self._build_configuration_snapshot(
                sequence_config,
                analysis_config,
            )
            if not transaction.call_owned(
                "project effective active sequence configuration",
                self.model.apply_configuration,
                effective_snapshot,
                generation=transaction.checkpoint.configuration_generation,
            ):
                return fail()
            transaction.mark_surface_verified("runtime")
            if not transaction.attempt(
                "mark configuration loaded",
                self.view.mark_configuration_loaded,
                surface="view",
            ):
                return fail()
        if not self._project_analysis_flags(transaction):
            return fail()
        if present:
            if not transaction.attempt(
                "present configuration availability",
                self.present_configuration_availability,
                configuration_available,
                rollback=lambda: self.present_configuration_availability(
                    bool(transaction.checkpoint.sequence_config)
                ),
                surface="view",
            ):
                return fail()
        if not configuration_available:
            if not transaction.attempt(
                "present missing configuration prompt",
                self.view.present_missing_configuration_prompt,
                self.model.sequence_config,
                eligible=False,
                surface="view",
            ):
                return fail()
        else:
            if not transaction.attempt(
                "update analysis configuration presentation",
                self._analysis_config_changed,
                self.model.analysis_config,
                rollback=lambda: self._analysis_config_changed(
                    self.model.analysis_config
                ),
                surface="analysis",
                verifies_surface=False,
            ):
                return fail()
            if not transaction.attempt(
                "refresh test mode availability",
                self._refresh_test_mode_availability,
                rollback=self._refresh_test_mode_availability,
                surface="mode_availability",
            ):
                return fail()

        candidate_event = ConfigurationChanged(
            configuration_generation=(
                transaction.checkpoint.configuration_generation + 1
            ),
            configuration_snapshot=self.model.current_snapshot(),
        )
        if not transaction.call_owned(
            "commit active sequence configuration model",
            self.model.apply_configuration,
            candidate_event.configuration_snapshot,
            generation=candidate_event.configuration_generation,
        ):
            return fail()
        transaction.mark_surface_verified("configuration")
        if regenerated and not self._attempt_config_persistence(
            transaction,
            sequence_config,
            self.model.using_config_path,
            semantic_checkpoint=persisted_config_checkpoint,
        ):
            self._abort_after_durable_failure(
                transaction, "regenerated sequence config persistence failed"
            )
            return None
        self._finish_configuration_transaction(transaction, candidate_event)
        return None

    def present_configuration_availability(self, available: bool) -> None:
        available = bool(available and self._actions_may_be_enabled())
        self.view.set_sequence_config_available(
            available,
            mode=self.model.acquisition_mode if available else None,
        )

    def runtime_action_readiness(
        self, mode: str | None = None
    ) -> tuple[bool, bool]:
        mode = mode or self.model.acquisition_mode
        data_struct = self.model.data_struct
        has_data = self._has_runtime_samples(
            getattr(data_struct, "store_wave_data", None)
        ) or self._has_runtime_samples(
            getattr(data_struct, "store_wave_data_multi", None)
        )
        data_ready = has_data
        if mode == "IMPORT_STIMULUS_AUDIO":
            data_ready = (
                self.has_imported_recording_runtime_state()
                and self.has_import_stimulus_runtime_reference()
            )
        replay_ready = has_data and mode != "IMPORT_AUDIO"
        return bool(replay_ready), bool(data_ready)

    def _project_analysis_flags(
        self, transaction: _ConfigurationTransaction
    ) -> bool:
        service = self._analysis_flag_projection_service
        try:
            checkpoint = service.capture_runtime_state()
            frozen_analysis_config = self.model.current_snapshot().analysis_config
        except Exception as exc:
            self._logger.warning(
                f"Failed to prepare analysis flag projection: {exc}"
            )
            return False
        return transaction.attempt(
            "project analysis flags",
            service.project,
            frozen_analysis_config,
            rollback=lambda: service.restore_runtime_state(checkpoint),
            required_projection=True,
            surface="analysis",
            verifies_surface=False,
        )

    @pyqtSlot(object)
    def handle_configuration_changed(self, event: ConfigurationChanged) -> bool:
        if type(event) is not ConfigurationChanged:
            return False
        if event.configuration_generation <= self.model.configuration_generation:
            return False
        snapshot = event.configuration_snapshot
        if type(snapshot) is not ConfigurationSnapshot:
            return False
        model_checkpoint = self.model.checkpoint_configuration_state()
        service = self._analysis_flag_projection_service
        try:
            flag_checkpoint = service.capture_runtime_state()
        except Exception as exc:
            self._logger.warning(
                f"Failed to prepare analysis flag projection: {exc}"
            )
            return False
        if not self.model.apply_configuration(
            snapshot, generation=event.configuration_generation
        ):
            return False
        try:
            service.project(snapshot.analysis_config)
        except BaseException as error:
            return self._recover_configuration_changed_projection(
                model_checkpoint,
                flag_checkpoint,
                error,
            )
        return True

    def _recover_configuration_changed_projection(
        self,
        model_checkpoint: Any,
        flag_checkpoint: Any,
        primary_error: BaseException,
    ) -> bool:
        failures = _RecoveryFailureAggregator(
            primary_error,
            operation="analysis flag projection",
        )
        restored = True
        for operation, callback, checkpoint in (
            (
                "configuration model restoration",
                self.model.restore_configuration_state,
                model_checkpoint,
            ),
            (
                "analysis flag restoration",
                self._analysis_flag_projection_service.restore_runtime_state,
                flag_checkpoint,
            ),
        ):
            try:
                callback(checkpoint)
            except BaseException as restore_error:
                restored = False
                failures.capture(operation, restore_error)
        if not restored:
            self._record_projection_failure(
                "analysis",
                "configuration event analysis flag rollback was incomplete",
            )
            try:
                self._disable_inconsistent_configuration()
            except BaseException as disable_error:
                failures.capture(
                    "disable inconsistent configuration",
                    disable_error,
                )
        failures.warning(
            self._logger,
            "Failed to project analysis flags: "
            f"{_safe_exception_description(primary_error)}",
            operation="analysis flag projection diagnostic logging",
        )
        if isinstance(primary_error, Exception):
            failures.raise_if_interrupted()
            return False
        failures.raise_if_selected()
        raise AssertionError("unreachable interruption recovery")

    def on_using_file_combobox_changed(self, text: str) -> None:
        if self._reject_reentrant_configuration_entry(
            "configuration selection"
        ):
            return None
        workflow_model = getattr(self.model, "_workflow_model", None)
        if workflow_model is not None and workflow_model.player_status_flag:
            self.restore_previous_configuration()
            self._warning("警告", "正在录音，请稍后...")
            return None
        transaction = self._begin_configuration_transaction(
            selection_path=self.model.using_config_path
        )
        if transaction is None:
            return None
        return self._run_owned_configuration_transaction(
            transaction,
            "configuration selection",
            self._on_using_file_combobox_changed_transaction,
            text,
        )

    def _on_using_file_combobox_changed_transaction(
        self,
        transaction: _ConfigurationTransaction,
        text: str,
    ) -> None:
        try:
            path = transaction.call_owned(
                "resolve selected configuration path",
                self.view.selected_path,
                text,
                self.model.registry,
            )
        except Exception as exc:
            self._logger.warning(
                f"Failed to resolve selected configuration path: {exc}"
            )
            transaction.abort()
            return None
        try:
            admitted_registry = dict(
                transaction.call_owned(
                    "capture active-path semantic checkpoint",
                    self._registry_loader,
                )
                or {}
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to capture active-path semantic checkpoint: "
                f"{exc}"
            )
            transaction.abort()
            return None
        using_path_semantic_checkpoint = _ActivePathSemanticCheckpoint(
            using_path_present="using_config_path" in admitted_registry,
            using_config_path=admitted_registry.get("using_config_path"),
            selected_key=text,
            selected_path_present=text in admitted_registry,
            selected_path=admitted_registry.get(text),
        )
        try:
            admitted_path_token = _canonical_json_semantic_token(
                using_path_semantic_checkpoint.selected_path
            )
            selected_path_token = _canonical_json_semantic_token(path)
        except TypeError as exc:
            self._logger.warning(
                f"Rejected active-path selection checkpoint: {exc}"
            )
            transaction.abort()
            return None
        if (
            not using_path_semantic_checkpoint.selected_path_present
            or admitted_path_token != selected_path_token
        ):
            self._logger.warning(
                "Rejected active-path selection: selected registry binding "
                "changed before admission"
            )
            transaction.abort()
            return None
        data_struct = self.model.data_struct

        def fail() -> None:
            transaction.abort()
            return None

        validated = transaction.call_owned(
            "load selected sequence configuration",
            self._load_validated_configuration,
            path,
        )
        configuration_available = validated is not None
        sequence_config, analysis_config = validated or ([], {})
        persisted_config_checkpoint = _copy_plain_configuration(sequence_config)
        provisional_snapshot = self._build_configuration_snapshot(
            sequence_config,
            analysis_config,
            using_config_path=path,
        )
        if not transaction.call_owned(
            "project selected configuration candidate",
            self.model.apply_configuration,
            provisional_snapshot,
            generation=transaction.checkpoint.configuration_generation,
        ):
            return fail()

        runtime_ready, regenerated = transaction.call_owned(
            "prepare selected configuration runtime",
            self._prepare_stimulus_config,
            sequence_config,
            path,
        )
        if not runtime_ready:
            return fail()
        effective_snapshot = self._build_configuration_snapshot(
            sequence_config,
            analysis_config,
            using_config_path=path,
        )
        if not transaction.call_owned(
            "project effective selected configuration",
            self.model.apply_configuration,
            effective_snapshot,
            generation=transaction.checkpoint.configuration_generation,
        ):
            return fail()

        if not self._project_analysis_flags(transaction):
            return fail()
        if not transaction.attempt(
            "update analysis configuration presentation",
            self._analysis_config_changed,
            self.model.analysis_config,
            rollback=lambda: self._analysis_config_changed(
                self.model.analysis_config
            ),
            surface="analysis",
            verifies_surface=False,
        ):
            return fail()
        if not transaction.attempt(
            "refresh test mode availability",
            self._refresh_test_mode_availability,
            rollback=self._refresh_test_mode_availability,
            surface="mode_availability",
        ):
            return fail()
        if not transaction.attempt(
            "refresh channel presentation",
            self._refresh_channels,
            rollback=self._refresh_channels,
            surface="channels",
        ):
            return fail()
        if not transaction.attempt(
            "reset configuration runtime actions",
            self.view.reset_runtime_action_buttons,
            surface="view",
        ):
            return fail()
        data_struct.store_wave_data = None
        data_struct.store_wave_data_multi = None
        self._clear_wav_calibration_runtime_state()
        transaction.mark_surface_verified("runtime")
        if not transaction.attempt(
            "present configuration availability",
            self.present_configuration_availability,
            configuration_available,
            rollback=lambda: self.present_configuration_availability(
                bool(transaction.checkpoint.sequence_config)
            ),
            surface="view",
        ):
            return fail()
        if configuration_available:
            if not transaction.attempt(
                "mark configuration loaded",
                self.view.mark_configuration_loaded,
                surface="view",
            ):
                return fail()
        else:
            if not transaction.attempt(
                "present missing configuration prompt",
                self.view.present_missing_configuration_prompt,
                self.model.sequence_config,
                eligible=False,
                surface="view",
            ):
                return fail()
        if not transaction.attempt(
            "focus after configuration selection",
            self.view.focus_after_selection,
            surface="view",
        ):
            return fail()

        candidate_event = ConfigurationChanged(
            configuration_generation=(
                transaction.checkpoint.configuration_generation + 1
            ),
            configuration_snapshot=self.model.current_snapshot(),
        )
        if not transaction.call_owned(
            "commit selected sequence configuration model",
            self.model.apply_configuration,
            candidate_event.configuration_snapshot,
            generation=candidate_event.configuration_generation,
        ):
            return fail()
        transaction.mark_surface_verified("configuration")
        if regenerated and not self._attempt_config_persistence(
            transaction,
            sequence_config,
            path,
            semantic_checkpoint=persisted_config_checkpoint,
        ):
            self._abort_after_durable_failure(
                transaction, "regenerated sequence config persistence failed"
            )
            return None
        path_persistence_result = self._attempt_using_path_persistence(
            transaction,
            path,
            semantic_checkpoint=using_path_semantic_checkpoint,
        )
        if not path_persistence_result:
            self._abort_after_durable_failure(
                transaction,
                "persist active sequence config path failed",
                persistence_surface="registry",
            )
            return None
        if not self._project_committed_active_path_registry(
            transaction,
            path_persistence_result,
            using_config_path=path,
        ):
            return None
        self._finish_verified_registry_projection_transaction(
            transaction, candidate_event
        )
        return None

    def restore_previous_configuration(self) -> None:
        if self.view.restore_selection(self.model.using_config_path):
            self._logger.warning("已恢复到之前的配置选项")
        self.present_configuration_availability(bool(self.model.sequence_config))
        return None

    def resolve_runtime_sample_rate(self, acq_mode: str, acq_detail: Any) -> Any:
        if acq_mode == "RECORD_ONLY":
            normalized_detail = normalize_record_only_detail(acq_detail)
            if normalized_detail.get("monitor_playback", False):
                return self._resolve_duplex_sample_rate(
                    self.model.mic, self.model.speaker
                )
            return self._resolve_input_sample_rate(self.model.mic)
        if acq_mode == "PLAY_AND_RECORD":
            return self._resolve_duplex_sample_rate(
                self.model.mic, self.model.speaker
            )
        return None

    @staticmethod
    def _is_positive_runtime_integer(value: Any) -> bool:
        return (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            and int(value) > 0
        )

    @staticmethod
    def _has_runtime_samples(value: Any) -> bool:
        if value is None:
            return False
        try:
            return np.asarray(value).size > 0
        except (TypeError, ValueError):
            return False

    def has_imported_recording_runtime_state(self) -> bool:
        data_struct = self.model.data_struct
        return (
            self._is_positive_runtime_integer(
                getattr(data_struct, "sample_rate", None)
            )
            and self._is_positive_runtime_integer(
                getattr(data_struct, "audio_lenth", None)
            )
            and self._has_runtime_samples(
                getattr(data_struct, "store_wave_data", None)
            )
            and self._has_runtime_samples(
                getattr(data_struct, "store_wave_data_multi", None)
            )
        )

    def has_import_stimulus_runtime_reference(self) -> bool:
        data_struct = self.model.data_struct
        stimulus_info = getattr(data_struct, "stimulus_info", None)
        if not isinstance(stimulus_info, dict):
            self.model.stimulus_reference_ready = False
            return False
        recording_rate = getattr(data_struct, "sample_rate", None)
        reference_rate = stimulus_info.get("sample_rate")
        ready = (
            self._is_positive_runtime_integer(recording_rate)
            and self._is_positive_runtime_integer(reference_rate)
            and int(recording_rate) == int(reference_rate)
            and self._has_runtime_samples(
                getattr(data_struct, "stimulus_data", None)
            )
        )
        self.model.stimulus_reference_ready = bool(ready)
        return bool(ready)

    def validate_import_stimulus_analysis_readiness(self) -> bool:
        readiness_message = (
            "分析参考激励尚未就绪或采样率与导入音频不一致，请检查激励配置后重试。"
        )
        if not (
            self.has_imported_recording_runtime_state()
            and self.has_import_stimulus_runtime_reference()
        ):
            self._warning("提示", readiness_message)
            return False
        data_struct = self.model.data_struct
        stimulus_info = data_struct.stimulus_info
        total_time = stimulus_info.get("total_time")
        if isinstance(total_time, (bool, np.bool_)) or not isinstance(
            total_time, (int, float, np.integer, np.floating)
        ):
            self._warning("提示", readiness_message)
            return False
        total_time = float(total_time)
        if not np.isfinite(total_time) or total_time <= 0:
            self._warning("提示", readiness_message)
            return False
        stimulus_length = round(total_time * int(stimulus_info["sample_rate"]))
        if int(data_struct.audio_lenth) != stimulus_length:
            self._warning(
                "音频长度校验失败",
                f"导入音频长度({data_struct.audio_lenth})\n"
                f"与激励信号长度({stimulus_length})不一致！无法分析！",
            )
            return False
        return True

    def _clear_stimulus_runtime_state(self, *, clear_sample_rate: bool) -> None:
        data_struct = self.model.data_struct
        if clear_sample_rate:
            data_struct.sample_rate = None
        data_struct.stimulus_data = None
        data_struct.stimulus_info = None
        if hasattr(data_struct, "alignment_sample_count"):
            delattr(data_struct, "alignment_sample_count")
        self.model.stimulus_reference_ready = False

    def _clear_wav_calibration_runtime_state(self) -> None:
        data_struct = self.model.data_struct
        data_struct.wav_calibration_metadata = None
        data_struct.wav_calibration_metadata_authoritative = False
        data_struct.wav_calibration_warning_shown = False

    def _invoke_callback(
        self,
        operation: str,
        callback: Callable[..., Any] | None,
        *args: Any,
        reject_false: bool = False,
        **kwargs: Any,
    ) -> bool:
        if callback is None:
            return True
        try:
            result = callback(*args, **kwargs)
        except Exception as exc:
            self._logger.warning(f"Failed to {operation}: {exc}")
            return False
        if reject_false and result is False:
            self._logger.warning(f"Failed to {operation}: callback rejected operation")
            return False
        return True

    def _set_data_enabled(self, enabled: bool) -> bool:
        return self._invoke_callback(
            "update analysis action availability",
            self._data_enabled_setter,
            bool(enabled),
        )

    def _clear_import_analysis_runtime_state(self, *, clear_plot: bool) -> bool:
        self._clear_import_runtime_data()
        if not self._invoke_callback(
            "clear retained import identity", self._clear_import_identity
        ):
            return False
        if not self._set_data_enabled(False):
            return False
        if clear_plot and not self._invoke_callback(
            "clear plot presentation", self._clear_plot
        ):
            return False
        return True

    def _clear_import_runtime_data(self) -> None:
        data_struct = self.model.data_struct
        data_struct.store_wave_data = None
        data_struct.store_wave_data_multi = None
        data_struct.sample_rate = None
        data_struct.audio_lenth = None
        self._clear_stimulus_runtime_state(clear_sample_rate=False)
        self._clear_wav_calibration_runtime_state()

    def refresh_import_stimulus_analysis_reference(self, acq_detail: Any) -> bool:
        if not self.has_imported_recording_runtime_state():
            self._set_data_enabled(False)
            self.model.stimulus_reference_ready = False
            return False
        data_struct = self.model.data_struct
        runtime_sample_rate = int(data_struct.sample_rate)
        staged_reference = self._stage_data_struct(data_struct)
        staged_reference.sample_rate = runtime_sample_rate
        try:
            reference_ready = self._analysis_reference_setter(
                staged_reference,
                acq_detail,
                using_config_path=self.model.using_config_path,
                runtime_sample_rate=runtime_sample_rate,
                logger=self._logger,
            )
        except Exception as exc:
            self._clear_stimulus_runtime_state(clear_sample_rate=False)
            self._set_data_enabled(False)
            self._warning("提示", f"加载分析参考激励失败: {str(exc)[:200]}")
            return False
        if not reference_ready:
            self._clear_stimulus_runtime_state(clear_sample_rate=False)
            self._set_data_enabled(False)
            self._warning("提示", "加载分析参考激励失败，请检查激励配置。")
            return False
        self._commit_staged_data_struct(data_struct, staged_reference)
        data_struct.sample_rate = runtime_sample_rate
        self.model.stimulus_reference_ready = True
        return self._set_data_enabled(True)

    def _prepare_stimulus_config(
        self,
        sequence_config: list[Any],
        using_config_path: Any,
    ) -> tuple[bool, bool]:
        if not sequence_config:
            return True, False
        acq_config = sequence_config[0]["seq1"]["acq"]
        mode = acq_config["mode"]
        detail = acq_config["detail"]
        if mode in ("IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"):
            self._clear_stimulus_runtime_state(
                clear_sample_rate=not self.has_imported_recording_runtime_state()
            )
            self.model.streaming_stimulus_data = None
            return True, False
        try:
            result = self.resolve_runtime_sample_rate(mode, detail)
        except Exception as exc:
            self._logger.warning(f"Failed to resolve runtime sample rate: {exc}")
            self._clear_stimulus_runtime_state(clear_sample_rate=True)
            self.model.streaming_stimulus_data = None
            return False, False
        if result is not None and not result.ok:
            self._logger.warning(result.message)
            self._clear_stimulus_runtime_state(clear_sample_rate=True)
            self.model.streaming_stimulus_data = None
            return True, False
        if result is not None:
            self.model.runtime_sample_rate = result.sample_rate
        if mode == "PLAY_AND_RECORD":
            if result is None:
                self._logger.warning(
                    "Failed to set runtime stimulus: runtime sample rate is unavailable"
                )
                self._clear_stimulus_runtime_state(clear_sample_rate=True)
                self.model.streaming_stimulus_data = None
                return False, False
            try:
                staged_data_struct = self._stage_data_struct(
                    self.model.data_struct
                )
                modified = self._stimulus_setter(
                    staged_data_struct,
                    detail,
                    using_config_path=using_config_path,
                    runtime_sample_rate=result.sample_rate,
                )
            except Exception as exc:
                self._clear_stimulus_runtime_state(clear_sample_rate=True)
                self.model.streaming_stimulus_data = None
                self._logger.warning(f"Failed to set runtime stimulus: {exc}")
                return False, False
            self._commit_staged_data_struct(
                self.model.data_struct, staged_data_struct
            )
            self.model.stimulus_reference_ready = self._has_runtime_samples(
                getattr(self.model.data_struct, "stimulus_data", None)
            )
            return True, bool(modified)
        self._clear_stimulus_runtime_state(clear_sample_rate=False)
        return True, False

    def init_data_struct_stimulus_config(self) -> None:
        if self._reject_reentrant_configuration_entry(
            "configuration runtime initialization"
        ):
            return None
        transaction = self._begin_configuration_transaction(
            selection_path=self.model.using_config_path
        )
        if transaction is None:
            return None
        return self._run_owned_configuration_transaction(
            transaction,
            "configuration runtime initialization",
            self._init_data_struct_stimulus_config_transaction,
        )

    def _init_data_struct_stimulus_config_transaction(
        self,
        transaction: _ConfigurationTransaction,
    ) -> None:
        sequence_config = _copy_plain_configuration(self.model.sequence_config)
        persisted_config_checkpoint = _copy_plain_configuration(sequence_config)
        prepared, regenerated = transaction.call_owned(
            "prepare initialized configuration runtime",
            self._prepare_stimulus_config,
            sequence_config,
            self.model.using_config_path,
        )
        if not prepared:
            transaction.abort()
            return None
        if not regenerated:
            transaction.commit_failure_provenance()
            return None
        effective_snapshot = self._build_configuration_snapshot(
            sequence_config,
            self.model.analysis_config,
        )
        event = ConfigurationChanged(
            configuration_generation=(
                transaction.checkpoint.configuration_generation + 1
            ),
            configuration_snapshot=effective_snapshot,
        )
        if not transaction.call_owned(
            "commit initialized sequence configuration model",
            self.model.apply_configuration,
            event.configuration_snapshot,
            generation=event.configuration_generation,
        ):
            transaction.abort()
            return None
        if not self._attempt_config_persistence(
            transaction,
            sequence_config,
            self.model.using_config_path,
            semantic_checkpoint=persisted_config_checkpoint,
        ):
            self._abort_after_durable_failure(
                transaction, "regenerated sequence config persistence failed"
            )
            return None
        self._finish_configuration_transaction(transaction, event)
        return None

    @staticmethod
    def _capture_data_struct_state(
        data_struct: Any,
    ) -> dict[str, _CapturedRuntimeField]:
        runtime = vars(data_struct)
        return {
            name: _CapturedRuntimeField(
                existed=name in runtime,
                value=runtime.get(name),
            )
            for name in _TRANSACTION_RUNTIME_FIELDS
        }

    @staticmethod
    def _restore_data_struct_state(
        data_struct: Any, state: dict[str, _CapturedRuntimeField]
    ) -> bool:
        target = vars(data_struct)
        for name, field in state.items():
            if field.existed:
                target[name] = field.value
            else:
                target.pop(name, None)
        return True

    @staticmethod
    def _stage_data_struct(data_struct: Any) -> SimpleNamespace:
        runtime = vars(data_struct)
        staged = SimpleNamespace(
            stimulus_data=None,
            stimulus_info=None,
            sample_rate=runtime.get("sample_rate"),
            alignment_sample_count=runtime.get("alignment_sample_count"),
        )
        return staged

    @staticmethod
    def _commit_staged_data_struct(
        data_struct: Any, staged_data_struct: Any
    ) -> None:
        target = vars(data_struct)
        staged = vars(staged_data_struct)
        for name in _STIMULUS_RUNTIME_FIELDS:
            if name in staged:
                target[name] = staged[name]
            else:
                target.pop(name, None)

    def _clear_known_runtime_data(self) -> None:
        runtime = vars(self.model.data_struct)
        runtime.update(_CLEAR_DATA_RUNTIME_DEFAULTS)

    def _prepare_candidate_runtime(
        self,
        *,
        transaction: _ConfigurationTransaction,
        old_acq: dict[Any, Any] | None,
        new_acq: dict[Any, Any],
        sequence_config: list[Any],
        using_config_path: Any,
    ) -> _RuntimePreparation:
        old_mode = old_acq.get("mode") if old_acq else None
        old_detail = old_acq.get("detail") if old_acq else None
        new_mode = new_acq["mode"]
        new_detail = new_acq["detail"]
        mode_changed = old_mode != new_mode
        if mode_changed:
            try:
                plot_state = transaction.capture_plot_state()
                import_modes = {"IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"}
                import_identity_state = (
                    transaction.capture_import_identity_state()
                    if old_mode in import_modes or new_mode in import_modes
                    else None
                )
            except Exception as exc:
                self._logger.warning(
                    f"Failed to prepare configuration cleanup projections: {exc}"
                )
                return _RuntimePreparation(failure="cleanup")
            transaction.defer_commit(
                "clear plot presentation",
                self._clear_plot,
                rollback=self._projection_restorer(
                    self._plot_state_restorer, plot_state
                ),
                surface="plot_import",
            )
            self._clear_known_runtime_data()
            if old_mode in import_modes or new_mode in import_modes:
                self._clear_import_runtime_data()
                old_data_enabled = transaction.view_state.action_availability[2]
                if not transaction.attempt(
                    "update analysis action availability",
                    self._data_enabled_setter,
                    False,
                    rollback=(
                        (lambda: self._data_enabled_setter(bool(old_data_enabled)))
                        if old_data_enabled is not None
                        else None
                    ),
                    surface="view",
                ):
                    return _RuntimePreparation(failure="cleanup")
                transaction.defer_commit(
                    "clear retained import identity",
                    self._clear_import_identity,
                    rollback=self._projection_restorer(
                        self._import_identity_state_restorer,
                        import_identity_state,
                    ),
                    surface="plot_import",
                )

        preserve_import_runtime = (
            not mode_changed and new_mode == "IMPORT_STIMULUS_AUDIO"
        )
        if preserve_import_runtime:
            if old_detail != new_detail:
                if self.has_imported_recording_runtime_state():
                    old_data_enabled = transaction.view_state.action_availability[2]
                    invoked, reference_ready = transaction.attempt_result(
                        "refresh import stimulus reference",
                        self.refresh_import_stimulus_analysis_reference,
                        new_detail,
                        rollback=(
                            (lambda: self._data_enabled_setter(bool(old_data_enabled)))
                            if old_data_enabled is not None
                            else None
                        ),
                        surface="runtime",
                        verifies_surface=False,
                    )
                    if not invoked or not reference_ready:
                        self._logger.warning(
                            "Failed to refresh import stimulus reference: "
                            "reference is unavailable"
                        )
                        return _RuntimePreparation(failure="reference")
                    return _RuntimePreparation(
                        verified_surfaces=frozenset({"runtime"})
                    )
                else:
                    try:
                        import_identity_state = (
                            transaction.capture_import_identity_state()
                        )
                        plot_state = transaction.capture_plot_state()
                    except Exception as exc:
                        self._logger.warning(
                            "Failed to prepare import cleanup projections: "
                            f"{exc}"
                        )
                        return _RuntimePreparation(failure="cleanup")
                    self._clear_import_runtime_data()
                    old_data_enabled = transaction.view_state.action_availability[2]
                    if not transaction.attempt(
                        "update analysis action availability",
                        self._data_enabled_setter,
                        False,
                        rollback=(
                            (
                                lambda: self._data_enabled_setter(
                                    bool(old_data_enabled)
                                )
                            )
                            if old_data_enabled is not None
                            else None
                        ),
                        surface="view",
                    ):
                        return _RuntimePreparation(failure="cleanup")
                    transaction.defer_commit(
                        "clear retained import identity",
                        self._clear_import_identity,
                        rollback=self._projection_restorer(
                            self._import_identity_state_restorer,
                            import_identity_state,
                        ),
                        surface="plot_import",
                    )
                    transaction.defer_commit(
                        "clear plot presentation",
                        self._clear_plot,
                        rollback=self._projection_restorer(
                            self._plot_state_restorer, plot_state
                        ),
                        surface="plot_import",
                    )
                    return _RuntimePreparation(
                        verified_surfaces=frozenset({"runtime"})
                    )
            elif not (
                self.has_imported_recording_runtime_state()
                and self.has_import_stimulus_runtime_reference()
            ):
                old_data_enabled = transaction.view_state.action_availability[2]
                if not transaction.attempt(
                    "update analysis action availability",
                    self._data_enabled_setter,
                    False,
                    rollback=(
                        (lambda: self._data_enabled_setter(bool(old_data_enabled)))
                        if old_data_enabled is not None
                        else None
                    ),
                    surface="view",
                ):
                    return _RuntimePreparation(failure="cleanup")
            return _RuntimePreparation()

        prepared, regenerated = self._prepare_stimulus_config(
            sequence_config, using_config_path
        )
        if not prepared:
            return _RuntimePreparation(failure="reference")
        return _RuntimePreparation(
            regenerated=regenerated,
            verified_surfaces=frozenset({"runtime"}),
        )

    def on_sequence_config_updated(self, *_args: Any) -> bool:
        if self._reject_reentrant_configuration_entry(
            "configuration update"
        ):
            return False
        transaction = self._begin_configuration_transaction(
            selection_path=self.model.using_config_path
        )
        if transaction is None:
            return False
        return bool(
            self._run_owned_configuration_transaction(
                transaction,
                "configuration update",
                self._on_sequence_config_updated_transaction,
            )
        )

    def _on_sequence_config_updated_transaction(
        self,
        transaction: _ConfigurationTransaction,
    ) -> bool:
        old_acq = (
            self.model.sequence_config[0]["seq1"]["acq"]
            if self.model.sequence_config
            else None
        )

        def fail() -> bool:
            transaction.abort()
            return False

        (
            candidate_path,
            candidate_registry,
            candidate_entries,
            persistence_required,
            candidate_selected_key,
        ) = transaction.call_owned(
            "load updated configuration registry candidate",
            self._load_registry_candidate,
        )
        validated = transaction.call_owned(
            "load updated sequence configuration",
            self._load_validated_configuration,
            candidate_path,
        )
        if validated is None:
            self._logger.warning(
                "Failed to refresh sequence config after update: invalid configuration"
            )
            return fail()
        sequence_config, analysis_config = validated
        persisted_config_checkpoint = _copy_plain_configuration(sequence_config)
        if not transaction.attempt(
            "populate sequence configuration selection",
            self.view.populate_configuration_entries,
            candidate_entries,
            using_config_path=candidate_path,
            clear_first=True,
            surface="view",
        ):
            return fail()
        transaction.call_owned(
            "project updated configuration registry",
            self.model.replace_registry,
            candidate_registry,
            using_config_path=candidate_path,
            entries=candidate_entries,
        )
        transaction.mark_surface_verified("registry")
        new_acq = sequence_config[0]["seq1"]["acq"]
        provisional_snapshot = self._build_configuration_snapshot(
            sequence_config,
            analysis_config,
            using_config_path=candidate_path,
        )
        if not transaction.call_owned(
            "project updated configuration candidate",
            self.model.apply_configuration,
            provisional_snapshot,
            generation=transaction.checkpoint.configuration_generation,
        ):
            return fail()
        runtime_preparation = transaction.call_owned(
            "prepare updated configuration runtime",
            self._prepare_candidate_runtime,
            transaction=transaction,
            old_acq=old_acq,
            new_acq=new_acq,
            sequence_config=sequence_config,
            using_config_path=self.model.using_config_path,
        )
        if runtime_preparation.failure is not None:
            return fail()
        for surface in runtime_preparation.verified_surfaces:
            transaction.mark_surface_verified(surface)

        candidate_snapshot = self._build_configuration_snapshot(
            sequence_config,
            analysis_config,
            using_config_path=candidate_path,
        )
        if not transaction.call_owned(
            "project effective updated configuration",
            self.model.apply_configuration,
            candidate_snapshot,
            generation=transaction.checkpoint.configuration_generation,
        ):
            return fail()

        mode_changed = (
            old_acq is None or old_acq.get("mode") != new_acq["mode"]
        )
        if mode_changed and not transaction.attempt(
            "refresh channel presentation",
            self._refresh_channels,
            rollback=self._refresh_channels,
            surface="channels",
        ):
            return fail()
        if not self._project_analysis_flags(transaction):
            return fail()
        if not transaction.attempt(
            "update analysis configuration presentation",
            self._analysis_config_changed,
            self.model.analysis_config,
            rollback=lambda: self._analysis_config_changed(
                self.model.analysis_config
            ),
            surface="analysis",
            verifies_surface=False,
        ):
            return fail()
        if self._analysis_config_changed is not None:
            transaction.mark_surface_verified("analysis")
        if not transaction.attempt(
            "refresh test mode availability",
            self._refresh_test_mode_availability,
            rollback=self._refresh_test_mode_availability,
            surface="mode_availability",
        ):
            return fail()
        if not transaction.attempt(
            "present configuration availability",
            self.present_configuration_availability,
            True,
            rollback=lambda: self.present_configuration_availability(
                bool(transaction.checkpoint.sequence_config)
            ),
            surface="view",
        ):
            return fail()

        if not transaction.attempt(
            "mark configuration loaded",
            self.view.mark_configuration_loaded,
            surface="view",
        ):
            return fail()
        candidate_event = ConfigurationChanged(
            configuration_generation=(
                transaction.checkpoint.configuration_generation + 1
            ),
            configuration_snapshot=self.model.current_snapshot(),
        )
        if not transaction.call_owned(
            "commit updated sequence configuration model",
            self.model.apply_configuration,
            candidate_event.configuration_snapshot,
            generation=candidate_event.configuration_generation,
        ):
            return fail()
        transaction.mark_surface_verified("configuration")
        if runtime_preparation.regenerated and not self._attempt_config_persistence(
            transaction,
            sequence_config,
            candidate_path,
            semantic_checkpoint=persisted_config_checkpoint,
        ):
            return self._abort_after_durable_failure(
                transaction, "regenerated sequence config persistence failed"
            )
        path_persistence_result = None
        if persistence_required:
            path_persistence_result = self._attempt_using_path_persistence(
                transaction,
                candidate_path,
                semantic_checkpoint=_ActivePathSemanticCheckpoint(
                    using_path_present=(
                        "using_config_path" in candidate_registry
                    ),
                    using_config_path=candidate_registry.get(
                        "using_config_path"
                    ),
                    selected_key=candidate_selected_key,
                    selected_path_present=(
                        candidate_selected_key in candidate_registry
                    ),
                    selected_path=candidate_registry.get(
                        candidate_selected_key
                    ),
                ),
            )
            if not path_persistence_result:
                return self._abort_after_durable_failure(
                    transaction,
                    "persist active sequence config path failed",
                    persistence_surface="registry",
                )
            if not self._project_committed_active_path_registry(
                transaction,
                path_persistence_result,
                using_config_path=candidate_path,
            ):
                return False
        return self._finish_configuration_transaction(
            transaction,
            candidate_event,
            registry_persistence_verified=(
                True if path_persistence_result is not None else None
            ),
        )

    def is_mode_available_for_external_trigger(self, mode: str | None = None) -> bool:
        return (mode or self.model.acquisition_mode) in {
            "RECORD_ONLY",
            "PLAY_AND_RECORD",
        }

    def can_output_ok_ng(self) -> tuple[bool, str]:
        config = self.model.analysis_config or {}
        sequence = config.get("display_sequence") or []
        if not isinstance(sequence, list) or not sequence:
            return False, "当前配置未选择任何分析项"
        for key in sequence:
            item = config.get(key)
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "AI" or (
                item_type in ("SPL", "SPLF", "FFT", "FR", "HD", "RB", "PRB")
                and item.get("limit_checked")
            ):
                return True, ""
        return False, "当前配置未启用阈值对比，无法产出OK/NG"

    def set_audio_devices_available(
        self, available: bool, message: str = ""
    ) -> None:
        self.model.set_audio_devices_available(available, message)
        self.view.refresh_availability()
        return None
