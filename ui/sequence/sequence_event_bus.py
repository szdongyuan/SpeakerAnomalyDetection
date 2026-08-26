"""Instance-scoped Qt command and event channels for one sequence window."""

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from pathlib import PosixPath, PurePosixPath, PureWindowsPath, WindowsPath
from threading import RLock
from weakref import WeakMethod, ref

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QObject, Qt, pyqtSignal

from ui.sequence.sequence_messages import (
    AnalysisExportPrepared,
    AnalysisExportPreparationFailed,
    AnalysisTransportReady,
    BeginRecordingRequested,
    CancelExportPreparationRequested,
    CancelRecordingRequested,
    CommitRecordingLabelRequested,
    ConfigurationSnapshot,
    ImportedAudioFailed,
    ImportedAudioReady,
    ExportPreparationCancelled,
    ManualLabelExportPrepared,
    ManualLabelExportPreparationFailed,
    PrepareAnalysisExportRequested,
    PrepareManualLabelExportRequested,
    ResourceLifecycleRequested,
    ShutdownFlushCompleted,
    ShutdownReady,
    WorkflowStateChanged,
    _FrozenMapping,
    _ImmutablePayloadArray,
    _validate_immutable_numpy_array,
)


_CONTINUATION_MESSAGE_CONTRACTS = {
    "export-preparation-cancel": (
        CancelExportPreparationRequested,
        ("request_id", "workflow_generation", "reason"),
    ),
    "export-preparation-cancelled": (
        ExportPreparationCancelled,
        ("request_id", "workflow_generation"),
    ),
    "analysis-export-prepare": (
        PrepareAnalysisExportRequested,
        (
            "request_id", "analysis_id", "source_id", "record_id",
            "workflow_generation", "result_snapshot", "analysis_configuration",
        ),
    ),
    "analysis-export-prepared": (
        AnalysisExportPrepared,
        (
            "request_id", "analysis_id", "source_id", "record_id",
            "workflow_generation", "result_snapshot", "target_configurations",
        ),
    ),
    "analysis-export-preparation-failed": (
        AnalysisExportPreparationFailed,
        (
            "request_id", "analysis_id", "source_id", "record_id",
            "workflow_generation", "reason",
        ),
    ),
    "manual-label-export-prepare": (
        PrepareManualLabelExportRequested,
        (
            "request_id", "command_id", "record_id", "label",
            "workflow_generation",
        ),
    ),
    "manual-label-export-prepared": (
        ManualLabelExportPrepared,
        (
            "request_id", "command_id", "record_id", "label",
            "workflow_generation", "result_snapshot", "target_configurations",
        ),
    ),
    "manual-label-export-preparation-failed": (
        ManualLabelExportPreparationFailed,
        (
            "request_id", "command_id", "record_id", "label",
            "workflow_generation", "reason",
        ),
    ),
    "workflow-state": (
        WorkflowStateChanged,
        (
            "workflow_generation",
            "previous_phase",
            "new_phase",
            "active_session_id",
            "active_import_id",
            "active_analysis_id",
            "active_job_id",
        ),
    ),
    "analysis-transport": (
        AnalysisTransportReady,
        (
            "analysis_id",
            "source_id",
            "record_id",
            "workflow_generation",
            "payload",
        ),
    ),
    "label-commit": (
        CommitRecordingLabelRequested,
        ("command_id", "record_id", "label", "export_outcome"),
    ),
    "shutdown-ready": (
        ShutdownReady,
        ("shutdown_generation",),
    ),
    "shutdown-flush-completed": (
        ShutdownFlushCompleted,
        ("shutdown_generation",),
    ),
}
_CONFIGURATION_SNAPSHOT_FIELDS = (
    "sequence_config",
    "analysis_config",
    "mic",
    "speaker",
    "mic_channels",
    "using_config_path",
    "streaming_stimulus_data",
)
_CANONICAL_IDENTITY_MAX_DEPTH = 8
_CANONICAL_IDENTITY_MAX_NODES = 8_192
_CANONICAL_IDENTITY_MAX_ITEMS = 512
_CANONICAL_IDENTITY_MAX_TEXT = 8_192
_CANONICAL_IDENTITY_MAX_ARRAY_BYTES = 268_435_456
_EXACT_PATH_TYPES = (PurePosixPath, PureWindowsPath, PosixPath, WindowsPath)


class _UnsupportedContinuationIdentity(Exception):
    pass


class ImportTerminalRecipientResult(Enum):
    """Exact acknowledgement semantics for import-terminal recipients."""

    ACK = auto()
    RETRYABLE_NACK = auto()
    PERMANENT_REJECT = auto()


class ResourceLifecycleRecipientResult(Enum):
    """Exact acknowledgement semantics for permanent lifecycle recipients."""

    ACK = auto()
    RETRYABLE_NACK = auto()


class WorkflowContinuationDeliveryStatus(Enum):
    """Exact publisher outcome for formal Workflow continuations."""

    ACK = auto()
    RETRYABLE_NACK = auto()
    PERMANENT_REJECT = auto()


class WorkflowContinuationRecipientResult(Enum):
    """Exact acknowledgement semantics for trusted continuation recipients."""

    ACK = auto()
    RETRYABLE_NACK = auto()
    PERMANENT_REJECT = auto()


@dataclass(frozen=True, slots=True)
class WorkflowContinuationDeliveryOutcome:
    status: WorkflowContinuationDeliveryStatus
    reason: str = ""


@dataclass(slots=True)
class _ContinuationIdentityBudget:
    nodes: int = 0
    array_bytes: int = 0


def _immutable_array_identity(value, budget):
    try:
        _validate_immutable_numpy_array(value)
        nbytes = int(np.ndarray.nbytes.__get__(value))
    except (BufferError, TypeError, ValueError) as error:
        raise _UnsupportedContinuationIdentity(
            "immutable array validation failed"
        ) from error
    budget.array_bytes += nbytes
    if budget.array_bytes > _CANONICAL_IDENTITY_MAX_ARRAY_BYTES:
        raise _UnsupportedContinuationIdentity("array byte budget exceeded")
    try:
        dtype = np.ndarray.dtype.__get__(value)
        shape = tuple(np.ndarray.shape.__get__(value))
        immutable_bytes = (
            b"" if nbytes == 0 else memoryview(value).cast("B")
        )
        digest = hashlib.blake2b(
            immutable_bytes,
            digest_size=32,
            person=b"seq-cont-v1",
        ).digest()
        dtype_bytes = np.dtype.str.__get__(dtype).encode("utf-8")
    except (BufferError, TypeError, ValueError) as error:
        raise _UnsupportedContinuationIdentity(
            "immutable array validation failed"
        ) from error
    return ("immutable-ndarray", dtype_bytes, shape, digest)


def _bounded_continuation_value_identity(value, depth, budget):
    if depth > _CANONICAL_IDENTITY_MAX_DEPTH:
        raise _UnsupportedContinuationIdentity("identity depth budget exceeded")
    budget.nodes += 1
    if budget.nodes > _CANONICAL_IDENTITY_MAX_NODES:
        raise _UnsupportedContinuationIdentity("identity node budget exceeded")
    value_type = type(value)
    if value_type is type(None):
        return ("none",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        if int.bit_length(value) > 4_096:
            raise _UnsupportedContinuationIdentity("integer bit budget exceeded")
        return ("int", value)
    if value_type is float:
        return ("float", float.hex(value))
    if value_type is complex:
        return ("complex", float.hex(value.real), float.hex(value.imag))
    if value_type is str:
        if str.__len__(value) > _CANONICAL_IDENTITY_MAX_TEXT:
            raise _UnsupportedContinuationIdentity("text budget exceeded")
        return ("str", value)
    if value_type is bytes:
        if bytes.__len__(value) > _CANONICAL_IDENTITY_MAX_TEXT:
            raise _UnsupportedContinuationIdentity("byte text budget exceeded")
        return ("bytes", value)
    if value_type is _ImmutablePayloadArray:
        return _immutable_array_identity(value, budget)
    if value_type is tuple:
        if tuple.__len__(value) > _CANONICAL_IDENTITY_MAX_ITEMS:
            raise _UnsupportedContinuationIdentity("tuple item budget exceeded")
        return (
            "tuple",
            tuple(
                _bounded_continuation_value_identity(item, depth + 1, budget)
                for item in tuple.__iter__(value)
            ),
        )
    if value_type is frozenset:
        if frozenset.__len__(value) > _CANONICAL_IDENTITY_MAX_ITEMS:
            raise _UnsupportedContinuationIdentity(
                "frozenset item budget exceeded"
            )
        return (
            "frozenset",
            frozenset(
                _bounded_continuation_value_identity(item, depth + 1, budget)
                for item in frozenset.__iter__(value)
            ),
        )
    if value_type is _FrozenMapping:
        items = object.__getattribute__(value, "_items")
        if (
            type(items) is not tuple
            or tuple.__len__(items) > _CANONICAL_IDENTITY_MAX_ITEMS
        ):
            raise _UnsupportedContinuationIdentity(
                "mapping item budget or storage validation failed"
            )
        canonical_items = []
        for pair in tuple.__iter__(items):
            if type(pair) is not tuple or tuple.__len__(pair) != 2:
                raise _UnsupportedContinuationIdentity(
                    "mapping pair validation failed"
                )
            canonical_items.append(
                (
                    _bounded_continuation_value_identity(
                        tuple.__getitem__(pair, 0), depth + 1, budget
                    ),
                    _bounded_continuation_value_identity(
                        tuple.__getitem__(pair, 1), depth + 1, budget
                    ),
                )
            )
        return ("mapping", frozenset(canonical_items))
    if value_type is ConfigurationSnapshot:
        return (
            "configuration-snapshot",
            tuple(
                (
                    field_name,
                    _bounded_continuation_value_identity(
                        object.__getattribute__(value, field_name),
                        depth + 1,
                        budget,
                    ),
                )
                for field_name in _CONFIGURATION_SNAPSHOT_FIELDS
            ),
        )
    if value_type is Decimal:
        decimal_tuple = Decimal.as_tuple(value)
        if len(decimal_tuple.digits) > _CANONICAL_IDENTITY_MAX_ITEMS:
            raise _UnsupportedContinuationIdentity("decimal digit budget exceeded")
        return (
            "decimal",
            decimal_tuple.sign,
            tuple(decimal_tuple.digits),
            decimal_tuple.exponent,
        )
    if value_type is datetime:
        timezone_identity = _bounded_continuation_value_identity(
            datetime.tzinfo.__get__(value), depth + 1, budget
        )
        return (
            "datetime",
            datetime.year.__get__(value),
            datetime.month.__get__(value),
            datetime.day.__get__(value),
            datetime.hour.__get__(value),
            datetime.minute.__get__(value),
            datetime.second.__get__(value),
            datetime.microsecond.__get__(value),
            datetime.fold.__get__(value),
            timezone_identity,
        )
    if value_type is date:
        return (
            "date",
            date.year.__get__(value),
            date.month.__get__(value),
            date.day.__get__(value),
        )
    if value_type is time:
        timezone_identity = _bounded_continuation_value_identity(
            time.tzinfo.__get__(value), depth + 1, budget
        )
        return (
            "time",
            time.hour.__get__(value),
            time.minute.__get__(value),
            time.second.__get__(value),
            time.microsecond.__get__(value),
            time.fold.__get__(value),
            timezone_identity,
        )
    if value_type is timedelta:
        return (
            "timedelta",
            timedelta.days.__get__(value),
            timedelta.seconds.__get__(value),
            timedelta.microseconds.__get__(value),
        )
    if value_type is timezone:
        offset = timezone.utcoffset(value, None)
        name = timezone.tzname(value, None)
        return (
            "timezone",
            _bounded_continuation_value_identity(
                offset, depth + 1, budget
            ),
            _bounded_continuation_value_identity(name, depth + 1, budget),
        )
    if value_type in _EXACT_PATH_TYPES:
        parts = PureWindowsPath.parts.__get__(value)
        if len(parts) > _CANONICAL_IDENTITY_MAX_ITEMS:
            raise _UnsupportedContinuationIdentity("path part budget exceeded")
        return (
            "path",
            value_type.__name__,
            tuple(
                _bounded_continuation_value_identity(
                    part, depth + 1, budget
                )
                for part in parts
            ),
        )
    raise _UnsupportedContinuationIdentity("unsupported identity value")


def _continuation_message_identity_outcome(kind, message):
    contract = _CONTINUATION_MESSAGE_CONTRACTS.get(kind)
    if contract is None:
        return None, "continuation kind is unsupported"
    if type(message) is not contract[0]:
        return None, "continuation message type is unsupported"
    try:
        budget = _ContinuationIdentityBudget()
        return (
            (
                contract[0].__name__,
                tuple(
                    (
                        field_name,
                        _bounded_continuation_value_identity(
                            object.__getattribute__(message, field_name),
                            0,
                            budget,
                        ),
                    )
                    for field_name in contract[1]
                ),
            ),
            "",
        )
    except _UnsupportedContinuationIdentity as error:
        return None, str(error)[:256]
    except BaseException:
        # This is the hostile-object boundary: malformed exact message instances
        # must not escape into publisher code or expose their payload in reasons.
        return None, "continuation message identity validation failed"


def _continuation_message_identity(kind, message):
    return _continuation_message_identity_outcome(kind, message)[0]


def _continuation_delivery_id(delivery_id, kind):
    try:
        if (
            type(delivery_id) is not tuple
            or not delivery_id
            or tuple.__len__(delivery_id) > 16
            or tuple.__getitem__(delivery_id, 0) != kind
        ):
            return None
        for item in tuple.__iter__(delivery_id):
            if type(item) is str:
                if not item or str.__len__(item) > 512:
                    return None
            elif type(item) is int:
                if int.bit_length(item) > 4_096:
                    return None
            else:
                return None
        return delivery_id
    except BaseException:
        return None


@dataclass(frozen=True, slots=True, eq=False)
class _ContinuationRecipientToken:
    """Opaque registration handle whose authority is its exact identity."""

    kind: str
    name: str
    version: int


@dataclass(frozen=True, slots=True)
class _ContinuationLifecycleOwnerToken:
    version: int


@dataclass(frozen=True, slots=True)
class _WorkflowContinuationAbandonment:
    delivery_id: tuple
    kind: str
    recipient_tokens: tuple[_ContinuationRecipientToken, ...]
    lifecycle_owner_token: _ContinuationLifecycleOwnerToken | None
    reason: str


@dataclass(frozen=True, slots=True)
class _WorkflowContinuationCompletion:
    lifecycle_owner_token: _ContinuationLifecycleOwnerToken
    delivery_id: tuple
    kind: str
    message_identity: tuple


@dataclass(frozen=True, slots=True)
class _ImportTerminalRecipientToken:
    name: str
    version: int
    critical: bool


@dataclass(frozen=True, slots=True, eq=False)
class _ResourceLifecycleRecipientToken:
    operation: str
    name: str
    version: int
    owner_identity: int | None


@dataclass(frozen=True, slots=True)
class _ResourceLifecycleRecipientRetirement:
    token: _ResourceLifecycleRecipientToken
    owner_ref: object
    reason: str


@dataclass(slots=True)
class _ResourceLifecycleDelivery:
    request: ResourceLifecycleRequested
    recipients: tuple[_ResourceLifecycleRecipientToken, ...]
    acknowledged: set[_ResourceLifecycleRecipientToken] = field(
        default_factory=set
    )
    in_progress: bool = False


@dataclass(frozen=True, slots=True)
class _ResourceLifecycleCompletion:
    request: ResourceLifecycleRequested


@dataclass(frozen=True, slots=True)
class _ResourceLifecycleAbandonment:
    shutdown_generation: int
    operation: str
    recipients: tuple[_ResourceLifecycleRecipientToken, ...]
    reason: str


@dataclass(slots=True)
class _ImportTerminalDelivery:
    delivery_id: tuple[str, str]
    message: ImportedAudioReady | ImportedAudioFailed
    recipients: tuple[_ImportTerminalRecipientToken, ...]
    acknowledged: set[_ImportTerminalRecipientToken] = field(
        default_factory=set
    )
    in_progress: bool = False


@dataclass(frozen=True, slots=True)
class _ImportTerminalCompletion:
    delivery_id: tuple[str, str]
    message: ImportedAudioReady | ImportedAudioFailed


class _GuardedImportTerminalRecipient:
    """Instance-tokened import recipient with optional QObject ownership."""

    def __init__(self, token, recipient, owner, retire) -> None:
        self.token = token
        self._active = True
        self._retire = retire
        self._strong_recipient = None
        self._weak_recipient = None
        self._owner_ref = None
        if owner is not None:
            if not isinstance(owner, QObject) or sip.isdeleted(owner):
                raise TypeError(
                    "import terminal recipient owner must be a live QObject"
                )
            self._owner_ref = ref(owner)
            owner.destroyed.connect(self.close)
        try:
            guard_ref = ref(self)

            def retire_collected_recipient(_recipient_ref):
                guard = guard_ref()
                if guard is not None:
                    guard.close()

            self._weak_recipient = WeakMethod(
                recipient, retire_collected_recipient
            )
        except TypeError:
            self._strong_recipient = recipient

    def close(self, *_args) -> None:
        if not self._active:
            return
        self._active = False
        self._retire(self.token)

    def matches_registration(self, recipient, owner) -> bool:
        return self.matches_recipient(recipient) and self.matches_owner(owner)

    def matches_recipient(self, recipient) -> bool:
        current = self._recipient()
        if current is None:
            return False
        current_self = getattr(current, "__self__", None)
        current_function = getattr(current, "__func__", None)
        recipient_self = getattr(recipient, "__self__", None)
        recipient_function = getattr(recipient, "__func__", None)
        if current_function is not None or recipient_function is not None:
            return (
                current_self is recipient_self
                and current_function is recipient_function
            )
        return current is recipient

    def matches_owner(self, owner) -> bool:
        if self._owner_ref is None:
            return owner is None
        return self._owner_ref() is owner

    def _recipient(self):
        if self._weak_recipient is not None:
            return self._weak_recipient()
        return self._strong_recipient

    def deliver(self, message) -> ImportTerminalRecipientResult:
        if not self._active:
            return ImportTerminalRecipientResult.RETRYABLE_NACK
        if self._owner_ref is not None:
            owner = self._owner_ref()
            if owner is None or sip.isdeleted(owner):
                self.close()
                return ImportTerminalRecipientResult.RETRYABLE_NACK
        recipient = self._recipient()
        if recipient is None:
            self.close()
            return ImportTerminalRecipientResult.RETRYABLE_NACK
        try:
            result = recipient(message)
        except BaseException:
            return ImportTerminalRecipientResult.RETRYABLE_NACK
        if type(result) is ImportTerminalRecipientResult:
            return result
        if result is True:
            return ImportTerminalRecipientResult.ACK
        # Legacy false/None results remain retryable: treating them as an ACK
        # or a permanent rejection would lose a critical Workflow delivery.
        return ImportTerminalRecipientResult.RETRYABLE_NACK


class _GuardedContinuationRecipient:
    """Direct Python recipient that becomes inert with its optional QObject owner."""

    def __init__(self, token, recipient, owner, retire) -> None:
        self.token = token
        self._active = True
        self._retire = retire
        self._strong_recipient = None
        self._weak_recipient = None
        self._owner_ref = None
        self._has_owner = owner is not None
        if owner is not None:
            if not isinstance(owner, QObject):
                raise TypeError("continuation recipient owner must be a QObject")
            self._owner_ref = ref(owner)
            owner.destroyed.connect(self.close)
        try:
            self._weak_recipient = WeakMethod(recipient)
        except TypeError:
            self._strong_recipient = recipient

    def close(self, *_args) -> None:
        if not self._active:
            return
        self._active = False
        self._retire(self.token)

    def matches_registration(self, recipient, owner) -> bool:
        return self.matches_recipient(recipient) and self.matches_owner(owner)

    def matches_recipient(self, recipient) -> bool:
        current = self._recipient()
        if current is None:
            return False
        current_self = getattr(current, "__self__", None)
        current_function = getattr(current, "__func__", None)
        recipient_self = getattr(recipient, "__self__", None)
        recipient_function = getattr(recipient, "__func__", None)
        if current_function is not None or recipient_function is not None:
            return (
                current_self is recipient_self
                and current_function is recipient_function
            )
        return current is recipient

    def matches_owner(self, owner) -> bool:
        if not self._has_owner:
            return owner is None
        return self._owner_ref is not None and self._owner_ref() is owner

    def _recipient(self):
        if self._weak_recipient is not None:
            return self._weak_recipient()
        return self._strong_recipient

    def deliver(self, message) -> WorkflowContinuationRecipientResult:
        if not self._active:
            return WorkflowContinuationRecipientResult.RETRYABLE_NACK
        if self._owner_ref is not None and self._owner_ref() is None:
            self.close()
            return WorkflowContinuationRecipientResult.RETRYABLE_NACK
        recipient = self._recipient()
        if recipient is None:
            self.close()
            return WorkflowContinuationRecipientResult.RETRYABLE_NACK
        try:
            result = recipient(message)
        except BaseException:
            return WorkflowContinuationRecipientResult.RETRYABLE_NACK
        if type(result) is WorkflowContinuationRecipientResult:
            return result
        # Projection handlers traditionally return None; only explicit False is
        # a negative acknowledgement.
        if result is False:
            return WorkflowContinuationRecipientResult.RETRYABLE_NACK
        return WorkflowContinuationRecipientResult.ACK


class _GuardedResourceLifecycleRecipient:
    """Exact lifecycle recipient guarded by callable and optional QObject life."""

    def __init__(self, token, recipient, owner, retire) -> None:
        self.token = token
        self._active = True
        self._retire = retire
        self._strong_recipient = None
        self._weak_recipient = None
        self._owner_ref = None
        self._has_owner = owner is not None
        if owner is not None:
            if not isinstance(owner, QObject) or sip.isdeleted(owner):
                raise TypeError(
                    "resource lifecycle recipient owner must be a live QObject"
                )
            self._owner_ref = ref(owner)
        try:
            self._weak_recipient = WeakMethod(recipient)
        except TypeError:
            self._strong_recipient = recipient

    def close(self, reason="explicit-retirement") -> None:
        if not self._active:
            return
        self._active = False
        self._retire(self.token, self._owner_ref, reason)

    def matches_registration(self, recipient, owner) -> bool:
        return self.matches_recipient(recipient) and self.matches_owner(owner)

    def matches_recipient(self, recipient) -> bool:
        current = self._recipient()
        if current is None:
            return False
        current_self = getattr(current, "__self__", None)
        current_function = getattr(current, "__func__", None)
        recipient_self = getattr(recipient, "__self__", None)
        recipient_function = getattr(recipient, "__func__", None)
        if current_function is not None or recipient_function is not None:
            return (
                current_self is recipient_self
                and current_function is recipient_function
            )
        return current is recipient

    def matches_owner(self, owner) -> bool:
        if not self._has_owner:
            return owner is None
        return self._owner_ref is not None and self._owner_ref() is owner

    def _recipient(self):
        if self._weak_recipient is not None:
            return self._weak_recipient()
        return self._strong_recipient

    def deliver(
        self, request: ResourceLifecycleRequested
    ) -> ResourceLifecycleRecipientResult:
        if not self._active:
            return ResourceLifecycleRecipientResult.RETRYABLE_NACK
        if self._owner_ref is not None:
            owner = self._owner_ref()
            try:
                owner_is_dead = owner is None or sip.isdeleted(owner)
            except BaseException:
                owner_is_dead = True
            if owner_is_dead:
                self.close("owner-destroyed")
                return ResourceLifecycleRecipientResult.RETRYABLE_NACK
        recipient = self._recipient()
        if recipient is None:
            self.close("recipient-destroyed")
            return ResourceLifecycleRecipientResult.RETRYABLE_NACK
        try:
            result = recipient(request)
        except BaseException:
            return ResourceLifecycleRecipientResult.RETRYABLE_NACK
        if type(result) is ResourceLifecycleRecipientResult:
            return result
        if result is False:
            return ResourceLifecycleRecipientResult.RETRYABLE_NACK
        return ResourceLifecycleRecipientResult.ACK


@dataclass(slots=True)
class _WorkflowContinuationDelivery:
    kind: str
    message: object
    message_identity: tuple
    recipients: tuple[_ContinuationRecipientToken, ...]
    lifecycle_owner_token: _ContinuationLifecycleOwnerToken
    acknowledged: set[_ContinuationRecipientToken] = field(default_factory=set)


class SequenceCommandChannels(QObject):
    start_test_requested = pyqtSignal(object)
    replay_requested = pyqtSignal(object)
    import_audio_requested = pyqtSignal(object)
    barcode_committed = pyqtSignal(object)
    manual_label_requested = pyqtSignal(object)
    manual_analysis_requested = pyqtSignal(object)

    begin_recording_requested = pyqtSignal(object)
    load_imported_audio_requested = pyqtSignal(object)
    analysis_requested = pyqtSignal(object)
    export_requested = pyqtSignal(object)
    commit_recording_label_requested = pyqtSignal(object)
    prepare_analysis_export_requested = pyqtSignal(object)
    prepare_manual_label_export_requested = pyqtSignal(object)
    cancel_export_preparation_requested = pyqtSignal(object)

    cancel_workflow_requested = pyqtSignal(object)
    cancel_imported_audio_requested = pyqtSignal(object)
    cancel_recording_requested = pyqtSignal(object)
    cancel_analysis_requested = pyqtSignal(object)
    cancel_export_requested = pyqtSignal(object)

    retry_export_requested = pyqtSignal(object)
    ignore_export_failure_requested = pyqtSignal(object)

    shutdown_requested = pyqtSignal(object)
    confirm_shutdown_cancellation_requested = pyqtSignal(object)
    abort_shutdown_requested = pyqtSignal(object)
    begin_shutdown_flush_requested = pyqtSignal(object)
    retry_shutdown_flush_requested = pyqtSignal(object)
    ignore_shutdown_flush_failure_requested = pyqtSignal(object)


class SequenceEventChannels(QObject):
    recording_started = pyqtSignal(object)
    recording_batch_ready = pyqtSignal(object)
    recording_completed = pyqtSignal(object)
    recording_failed = pyqtSignal(object)
    recording_cancelled = pyqtSignal(object)

    imported_audio_ready = pyqtSignal(object)
    imported_audio_failed = pyqtSignal(object)
    analysis_completed = pyqtSignal(object)
    analysis_failed = pyqtSignal(object)
    analysis_transport_ready = pyqtSignal(object)
    analysis_export_prepared = pyqtSignal(object)
    analysis_export_preparation_failed = pyqtSignal(object)
    manual_label_export_prepared = pyqtSignal(object)
    manual_label_export_preparation_failed = pyqtSignal(object)
    export_preparation_cancelled = pyqtSignal(object)
    export_completed = pyqtSignal(object)
    export_failed = pyqtSignal(object)
    export_retry_accepted = pyqtSignal(object)
    recording_label_committed = pyqtSignal(object)
    recording_label_commit_failed = pyqtSignal(object)

    workflow_command_rejected = pyqtSignal(object)
    workflow_state_changed = pyqtSignal(object)
    configuration_changed = pyqtSignal(object)
    shutdown_flush_failed = pyqtSignal(object)
    shutdown_flush_completed = pyqtSignal(object)
    shutdown_aborted = pyqtSignal(object)
    shutdown_ready = pyqtSignal(object)


class RetainedCleanupLifecycleRegistrationResult(Enum):
    """Exact outcome of retaining one native cleanup lifecycle root."""

    REGISTERED = auto()
    IDEMPOTENT = auto()
    TOKEN_COLLISION = auto()
    CLOSED_NATIVE_DELETED = auto()
    UNAVAILABLE = auto()


class _RetainedCleanupLifecycleRegistry:
    """Per-bus exact-token roots for native-finalization capsules."""

    def __init__(self) -> None:
        self._lock = RLock()
        # Key by ``id`` while retaining the token itself. This gives exact
        # object-identity semantics without invoking caller-defined hashing or
        # equality during teardown.
        self._entries: dict[int, tuple[object, object]] = {}
        self._closed = False

    def register(
        self, token: object, lifecycle: object
    ) -> RetainedCleanupLifecycleRegistrationResult:
        with self._lock:
            if self._closed:
                return (
                    RetainedCleanupLifecycleRegistrationResult.CLOSED_NATIVE_DELETED
                )
            existing = self._entries.get(id(token))
            if existing is not None:
                existing_token, existing_lifecycle = existing
                if existing_token is token and existing_lifecycle is lifecycle:
                    return (
                        RetainedCleanupLifecycleRegistrationResult.IDEMPOTENT
                    )
                return (
                    RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
                )
            self._entries[id(token)] = (token, lifecycle)
            return RetainedCleanupLifecycleRegistrationResult.REGISTERED

    def resolve(self, token: object) -> object | None:
        with self._lock:
            existing = self._entries.get(id(token))
            if existing is None or existing[0] is not token:
                return None
            return existing[1]

    def retire(self, token: object, lifecycle: object) -> bool:
        with self._lock:
            existing = self._entries.get(id(token))
            if (
                existing is None
                or existing[0] is not token
                or existing[1] is not lifecycle
            ):
                return False
            del self._entries[id(token)]
            return True

    def clear(self, *_destroyed) -> None:
        with self._lock:
            self._closed = True
            self._entries.clear()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)


_RECORDING_ADMISSION_CAPABILITY_KEY = object()


class _CanonicalRecordingAdmissionCapability:
    """Opaque identity-only authority; it cannot be copied or serialized."""

    __slots__ = ()

    def __new__(cls, key: object):
        if key is not _RECORDING_ADMISSION_CAPABILITY_KEY:
            raise TypeError("recording admission capability is unforgeable")
        return super().__new__(cls)

    def __copy__(self):
        raise TypeError("recording admission capability cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("recording admission capability cannot be copied")

    def __reduce_ex__(self, _protocol):
        raise TypeError("recording admission capability cannot be serialized")


@dataclass(frozen=True, slots=True)
class _CanonicalRecordingBeginClaim:
    identity: tuple[str, int]
    cancellation: CancelRecordingRequested | None = None
    workflow_admitted: bool = True


class _CanonicalRecordingAdmissionRegistry:
    """Capability-gated Workflow admission for one exact live session."""

    def __init__(self, *, standalone_recording_admission: bool = False) -> None:
        self._lock = RLock()
        self._command: BeginRecordingRequested | None = None
        self._identity: tuple[str, int] | None = None
        self._command_workflow_epoch: int | None = None
        self._workflow_epoch = 0
        self._workflow_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = None
        self._workflow_owner_ref = None
        self._standalone_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = (
            _CanonicalRecordingAdmissionCapability(
                _RECORDING_ADMISSION_CAPABILITY_KEY
            )
            if standalone_recording_admission
            else None
        )
        self._standalone_consumer_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = None
        self._consumer_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = None
        self._consumer_owner_ref = None
        self._standby_consumer_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = None
        self._standby_consumer_owner_ref = None
        self._claimed_consumer_capability: (
            _CanonicalRecordingAdmissionCapability | None
        ) = None
        self._cancellation: CancelRecordingRequested | None = None

    @staticmethod
    def _identity_for(
        command: object,
    ) -> tuple[str, int] | None:
        if type(command) is not BeginRecordingRequested:
            return None
        snapshot = command.session_snapshot
        if not isinstance(snapshot, Mapping):
            return None
        generation = snapshot.get("workflow_generation")
        if type(generation) is not int:
            return None
        return command.session_id, generation

    @staticmethod
    def _owner_is_live(owner_ref) -> bool:
        owner = None if owner_ref is None else owner_ref()
        return bool(owner is not None and not sip.isdeleted(owner))

    def _bind_owner(self, owner: object, *, workflow: bool):
        if not isinstance(owner, QObject) or sip.isdeleted(owner):
            return None
        capability_name = (
            "_workflow_capability" if workflow else "_consumer_capability"
        )
        owner_ref_name = (
            "_workflow_owner_ref" if workflow else "_consumer_owner_ref"
        )
        with self._lock:
            capability = getattr(self, capability_name)
            owner_ref = getattr(self, owner_ref_name)
            current_owner = None if owner_ref is None else owner_ref()
            if capability is not None and current_owner is owner:
                return capability
            if capability is not None and self._owner_is_live(owner_ref):
                return None
            if capability is not None:
                self._command = None
                self._identity = None
                self._command_workflow_epoch = None
            if workflow:
                # This monotonically growing epoch is also the permanent
                # tombstone that distinguishes a released Workflow owner from
                # a bus explicitly created for a standalone Recording harness.
                self._workflow_epoch += 1
                self._standalone_capability = None
                self._standalone_consumer_capability = None
                # A standalone consumer may already have atomically claimed a
                # command but not yet retired it into Recording preparation.
                # Establishing the first Workflow epoch revokes that in-flight
                # issuance under the same lock so its later retire cannot
                # authorize any side effect.
                self._command = None
                self._identity = None
                self._command_workflow_epoch = None
                self._claimed_consumer_capability = None
                self._cancellation = None
            capability = _CanonicalRecordingAdmissionCapability(
                _RECORDING_ADMISSION_CAPABILITY_KEY
            )
            setattr(self, capability_name, capability)
            setattr(self, owner_ref_name, ref(owner))
        release = self.release_workflow if workflow else self.release_consumer
        try:
            owner.destroyed.connect(
                lambda _destroyed=None, token=capability: release(token),
                Qt.QueuedConnection,
            )
        except (RuntimeError, TypeError):
            release(capability)
            return None
        if sip.isdeleted(owner):
            release(capability)
            return None
        return capability

    def bind_workflow_owner(self, owner: object):
        return self._bind_owner(owner, workflow=True)

    def bind_consumer(self, owner: object):
        if not isinstance(owner, QObject) or sip.isdeleted(owner):
            return None
        with self._lock:
            active_owner = (
                None
                if self._consumer_owner_ref is None
                else self._consumer_owner_ref()
            )
            if self._consumer_capability is not None and active_owner is owner:
                return self._consumer_capability
            if self._consumer_capability is not None and not self._owner_is_live(
                self._consumer_owner_ref
            ):
                if (
                    self._consumer_capability
                    is self._standalone_consumer_capability
                ):
                    self._standalone_capability = None
                    self._standalone_consumer_capability = None
                self._consumer_capability = None
                self._consumer_owner_ref = None
                self._claimed_consumer_capability = None
            if self._consumer_capability is None:
                capability = _CanonicalRecordingAdmissionCapability(
                    _RECORDING_ADMISSION_CAPABILITY_KEY
                )
                self._consumer_capability = capability
                self._consumer_owner_ref = ref(owner)
                if (
                    self._standalone_capability is not None
                    and self._workflow_epoch == 0
                ):
                    self._standalone_consumer_capability = capability
                return capability
            standby_owner = (
                None
                if self._standby_consumer_owner_ref is None
                else self._standby_consumer_owner_ref()
            )
            if (
                self._standby_consumer_capability is not None
                and standby_owner is owner
            ):
                return self._standby_consumer_capability
            if (
                self._standby_consumer_capability is not None
                and self._owner_is_live(self._standby_consumer_owner_ref)
            ):
                return None
            capability = _CanonicalRecordingAdmissionCapability(
                _RECORDING_ADMISSION_CAPABILITY_KEY
            )
            self._standby_consumer_capability = capability
            self._standby_consumer_owner_ref = ref(owner)
            return capability

    def register(self, capability: object, command: object) -> bool:
        identity = self._identity_for(command)
        if identity is None:
            return False
        with self._lock:
            if capability is not self._workflow_capability:
                return False
            if not self._owner_is_live(self._workflow_owner_ref):
                return False
            if self._command is command:
                return self._identity == identity
            if self._command is not None:
                return False
            self._command = command
            self._identity = identity
            self._command_workflow_epoch = self._workflow_epoch
            self._claimed_consumer_capability = None
            self._cancellation = None
            return True

    def claim_begin(
        self,
        capability: object,
        command: object,
    ) -> _CanonicalRecordingBeginClaim | None:
        identity = self._identity_for(command)
        with self._lock:
            if (
                self._standalone_capability is not None
                and self._workflow_epoch == 0
                and self._workflow_capability is None
                and self._command is None
                and capability is self._standalone_consumer_capability
                and capability is self._consumer_capability
                and self._owner_is_live(self._consumer_owner_ref)
                and identity is not None
            ):
                # Standalone Recording harnesses have no Workflow producer.
                # Canonicalize their exact command at this same atomic claim;
                # standby/capabilityless consumers remain unauthorized.
                self._command = command
                self._identity = identity
                self._command_workflow_epoch = None
                self._claimed_consumer_capability = capability
                self._cancellation = None
                return _CanonicalRecordingBeginClaim(
                    identity,
                    workflow_admitted=False,
                )
            if (
                capability is not self._consumer_capability
                or not self._owner_is_live(self._consumer_owner_ref)
                or self._claimed_consumer_capability is not None
                or identity is None
                or self._workflow_capability is None
                or not self._owner_is_live(self._workflow_owner_ref)
                or self._command_workflow_epoch != self._workflow_epoch
                or self._command is not command
                or self._identity != identity
            ):
                return None
            cancellation = self._cancellation
            if cancellation is not None:
                self._command = None
                self._identity = None
                self._command_workflow_epoch = None
                self._cancellation = None
                return _CanonicalRecordingBeginClaim(identity, cancellation)
            self._claimed_consumer_capability = capability
            return _CanonicalRecordingBeginClaim(identity)

    def claim_terminal(
        self,
        capability: object,
        command: object,
    ) -> tuple[str, int] | None:
        identity = self._identity_for(command)
        with self._lock:
            if (
                capability is not self._consumer_capability
                or not self._owner_is_live(self._consumer_owner_ref)
                or identity is None
                or self._command is not command
                or self._identity != identity
                or self._claimed_consumer_capability is not capability
            ):
                return None
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._claimed_consumer_capability = None
            self._cancellation = None
            return identity

    def retire(self, capability: object, command: object) -> bool:
        with self._lock:
            if (
                capability is not self._consumer_capability
                or not self._owner_is_live(self._consumer_owner_ref)
                or self._command is not command
                or self._claimed_consumer_capability is not capability
            ):
                return False
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._claimed_consumer_capability = None
            self._cancellation = None
            return True

    def retire_identity(
        self,
        capability: object,
        session_id: object,
        generation: object,
    ) -> bool:
        if (
            type(session_id) is not str
            or not session_id
            or type(generation) is not int
        ):
            return False
        with self._lock:
            if (
                capability is not self._workflow_capability
                or not self._owner_is_live(self._workflow_owner_ref)
                or self._identity != (session_id, generation)
            ):
                return False
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._claimed_consumer_capability = None
            self._cancellation = None
            return True

    def abandon(self, capability: object, command: object) -> bool:
        with self._lock:
            if (
                capability is not self._workflow_capability
                or not self._owner_is_live(self._workflow_owner_ref)
                or self._command is not command
            ):
                return False
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._claimed_consumer_capability = None
            self._cancellation = None
            return True

    def record_cancellation(
        self,
        capability: object,
        command: object,
    ) -> bool:
        if type(command) is not CancelRecordingRequested:
            return False
        identity = (command.session_id, command.workflow_generation)
        with self._lock:
            if (
                capability is not self._workflow_capability
                or not self._owner_is_live(self._workflow_owner_ref)
            ):
                return False
            if self._command is None:
                return True
            if self._identity != identity:
                return False
            if self._cancellation is None:
                self._cancellation = command
                return True
            return self._cancellation is command

    def replay(
        self,
        capability: object,
    ) -> BeginRecordingRequested | None:
        with self._lock:
            if (
                capability is not self._consumer_capability
                or not self._owner_is_live(self._consumer_owner_ref)
            ):
                return None
            if self._command is None:
                return None
            if self._command_workflow_epoch is None:
                if (
                    self._standalone_capability is None
                    or capability is not self._standalone_consumer_capability
                    or self._workflow_epoch != 0
                ):
                    return None
            elif (
                self._workflow_capability is None
                or not self._owner_is_live(self._workflow_owner_ref)
                or self._command_workflow_epoch != self._workflow_epoch
            ):
                return None
            return self._command

    def release_workflow(self, capability: object) -> bool:
        with self._lock:
            if capability is not self._workflow_capability:
                return False
            self._workflow_capability = None
            self._workflow_owner_ref = None
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._claimed_consumer_capability = None
            self._cancellation = None
            return True

    def release_consumer(
        self,
        capability: object,
        *,
        native_owner_destroyed: bool = False,
    ) -> tuple[bool, BeginRecordingRequested | None]:
        with self._lock:
            if capability is self._standby_consumer_capability:
                self._standby_consumer_capability = None
                self._standby_consumer_owner_ref = None
                return True, None
            if capability is not self._consumer_capability:
                return False, None
            if capability is self._standalone_consumer_capability:
                self._standalone_capability = None
                self._standalone_consumer_capability = None
                self._command = None
                self._identity = None
                self._command_workflow_epoch = None
                self._cancellation = None
            self._consumer_capability = None
            self._consumer_owner_ref = None
            self._claimed_consumer_capability = None
            standby_owner = (
                None
                if self._standby_consumer_owner_ref is None
                else self._standby_consumer_owner_ref()
            )
            if standby_owner is not None and (
                native_owner_destroyed or not sip.isdeleted(standby_owner)
            ):
                self._consumer_capability = self._standby_consumer_capability
                self._consumer_owner_ref = self._standby_consumer_owner_ref
                self._standby_consumer_capability = None
                self._standby_consumer_owner_ref = None
                return True, self._command
            self._standby_consumer_capability = None
            self._standby_consumer_owner_ref = None
            return True, None

    def clear(self, *_destroyed: object) -> None:
        with self._lock:
            self._command = None
            self._identity = None
            self._command_workflow_epoch = None
            self._workflow_capability = None
            self._workflow_owner_ref = None
            self._standalone_capability = None
            self._standalone_consumer_capability = None
            self._consumer_capability = None
            self._consumer_owner_ref = None
            self._standby_consumer_capability = None
            self._standby_consumer_owner_ref = None
            self._claimed_consumer_capability = None
            self._cancellation = None


class SequenceEventBus(QObject):
    """Own named command/event channels for exactly one sequence facade."""

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        standalone_recording_admission: bool = False,
    ) -> None:
        super().__init__(parent)
        if type(standalone_recording_admission) is not bool:
            raise TypeError("standalone_recording_admission must be a bool")
        self._retained_cleanup_lifecycle_registry = (
            _RetainedCleanupLifecycleRegistry()
        )
        self._canonical_recording_admission_registry = (
            _CanonicalRecordingAdmissionRegistry(
                standalone_recording_admission=(
                    standalone_recording_admission
                )
            )
        )
        self.commands = SequenceCommandChannels(self)
        self.events = SequenceEventChannels(self)
        self._export_retry_recipient = None
        self._export_terminal_recipients = OrderedDict()
        self._export_terminal_pending = {}
        self._export_terminal_completed = OrderedDict()
        self._export_terminal_history_limit = 128
        self._import_terminal_recipients = OrderedDict()
        self._import_terminal_recipient_version = 0
        self._import_terminal_recipient_limit = 32
        self._import_terminal_pending = OrderedDict()
        self._import_terminal_pending_limit = 128
        self._import_terminal_claims = {}
        self._import_terminal_capacity_waiting = None
        self._import_terminal_completed = OrderedDict()
        self._import_terminal_abandoned = OrderedDict()
        self._import_terminal_history_limit = 128
        self._import_terminal_dispatch_active = True
        self._workflow_continuation_recipients = {}
        self._workflow_continuation_recipient_limit = 32
        self._workflow_continuation_recipient_total_limit = 32
        self._workflow_continuation_recipient_version = 0
        self._workflow_continuation_lifecycle_owners = {}
        self._workflow_continuation_lifecycle_owner_version = 0
        self._workflow_continuation_lifecycle_owner_limit = 8
        self._workflow_continuation_pending = OrderedDict()
        self._workflow_continuation_active = set()
        self._workflow_continuation_completed = OrderedDict()
        self._workflow_continuation_abandoned = OrderedDict()
        self._workflow_continuation_history_limit = 128
        self._workflow_continuation_dispatch_active = True
        self._resource_lifecycle_recipients = {}
        self._resource_lifecycle_recipient_retirements = OrderedDict()
        self._resource_lifecycle_recipient_version = 0
        self._resource_lifecycle_recipient_limit = 32
        self._resource_lifecycle_recipient_total_limit = 32
        self._resource_lifecycle_pending = OrderedDict()
        self._resource_lifecycle_pending_limit = 32
        self._resource_lifecycle_completed = OrderedDict()
        self._resource_lifecycle_abandoned = OrderedDict()
        self._resource_lifecycle_history_limit = 128
        self._resource_lifecycle_dispatch_active = True
        self._resource_lifecycle_lock = RLock()
        self.destroyed.connect(
            self._retained_cleanup_lifecycle_registry.clear
        )
        self.destroyed.connect(
            self._canonical_recording_admission_registry.clear
        )
        self.destroyed.connect(self.close_import_terminal_dispatcher)
        self.destroyed.connect(self.close_workflow_continuation_dispatcher)
        self.destroyed.connect(self.close_resource_lifecycle_dispatcher)

    def _register_retained_cleanup_lifecycle(
        self, token: object, lifecycle: object
    ) -> RetainedCleanupLifecycleRegistrationResult:
        if sip.isdeleted(self):
            return (
                RetainedCleanupLifecycleRegistrationResult.CLOSED_NATIVE_DELETED
            )
        return self._retained_cleanup_lifecycle_registry.register(
            token, lifecycle
        )

    def _resolve_retained_cleanup_lifecycle(
        self, token: object
    ) -> object | None:
        return self._retained_cleanup_lifecycle_registry.resolve(token)

    def _retire_retained_cleanup_lifecycle(
        self, token: object, lifecycle: object
    ) -> bool:
        return self._retained_cleanup_lifecycle_registry.retire(
            token, lifecycle
        )

    def _retained_cleanup_lifecycle_count(self) -> int:
        return self._retained_cleanup_lifecycle_registry.active_count

    def _bind_canonical_recording_workflow_owner(self, owner: object):
        return self._canonical_recording_admission_registry.bind_workflow_owner(
            owner
        )

    def _bind_canonical_recording_consumer(self, owner: object):
        capability = self._canonical_recording_admission_registry.bind_consumer(
            owner
        )
        if capability is None:
            return None
        bus_ref = ref(self)
        registry = self._canonical_recording_admission_registry

        def release_consumer(*_destroyed):
            # QObject destruction can run while the controller/controller-bus
            # Python reference cycle is being cleared. Retire against the
            # plain registry first; only resolve the weak QObject wrapper if a
            # live standby actually needs its exact command replayed.
            released, replay = registry.release_consumer(
                capability,
                native_owner_destroyed=True,
            )
            if not released or replay is None:
                return
            owner_bus = bus_ref()
            if owner_bus is not None and not sip.isdeleted(owner_bus):
                owner_bus.commands.begin_recording_requested.emit(replay)

        try:
            owner.destroyed.connect(
                release_consumer,
                Qt.QueuedConnection,
            )
        except (AttributeError, RuntimeError, TypeError):
            self._release_canonical_recording_consumer(capability)
            return None
        if sip.isdeleted(owner):
            self._release_canonical_recording_consumer(capability)
            return None
        return capability

    def _register_canonical_recording_admission(
        self,
        capability: object,
        command: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.register(
            capability,
            command,
        )

    def _claim_canonical_recording_terminal(
        self,
        capability: object,
        command: object,
    ) -> tuple[str, int] | None:
        return self._canonical_recording_admission_registry.claim_terminal(
            capability,
            command,
        )

    def _claim_canonical_recording_begin(
        self,
        capability: object,
        command: object,
    ) -> _CanonicalRecordingBeginClaim | None:
        return self._canonical_recording_admission_registry.claim_begin(
            capability,
            command,
        )

    def _retire_canonical_recording_admission(
        self,
        capability: object,
        command: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.retire(
            capability,
            command,
        )

    def _retire_canonical_recording_identity(
        self,
        capability: object,
        session_id: object,
        generation: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.retire_identity(
            capability,
            session_id,
            generation,
        )

    def _abandon_canonical_recording_admission(
        self,
        capability: object,
        command: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.abandon(
            capability,
            command,
        )

    def _replay_canonical_recording_admission(
        self,
        capability: object,
    ) -> BeginRecordingRequested | None:
        return self._canonical_recording_admission_registry.replay(capability)

    def _record_canonical_recording_cancellation(
        self,
        capability: object,
        command: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.record_cancellation(
            capability,
            command,
        )

    def _release_canonical_recording_workflow_owner(
        self,
        capability: object,
    ) -> bool:
        return self._canonical_recording_admission_registry.release_workflow(
            capability
        )

    def _release_canonical_recording_consumer(
        self,
        capability: object,
    ) -> bool:
        released, replay = (
            self._canonical_recording_admission_registry.release_consumer(
                capability
            )
        )
        if replay is not None:
            self.commands.begin_recording_requested.emit(replay)
        return released

    def register_resource_lifecycle_recipient(
        self, operation, name, recipient, *, owner=None
    ):
        """Register one exact recipient for one permanent lifecycle operation."""
        if type(operation) is not str or not operation:
            raise ValueError("resource lifecycle operation must be non-empty")
        if type(name) is not str or not name or not callable(recipient):
            raise ValueError(
                "resource lifecycle recipient requires a name and callback"
            )
        if owner is not None and (
            not isinstance(owner, QObject) or sip.isdeleted(owner)
        ):
            raise TypeError(
                "resource lifecycle recipient owner must be a live QObject"
            )
        with self._resource_lifecycle_lock:
            recipients = self._resource_lifecycle_recipients.get(operation)
            current = None if recipients is None else recipients.get(name)
            if current is not None and current.matches_registration(
                recipient, owner
            ):
                return current.token
            total = sum(
                len(registered)
                for registered in self._resource_lifecycle_recipients.values()
            )
            if (
                current is None
                and total >= self._resource_lifecycle_recipient_total_limit
            ):
                raise RuntimeError("resource lifecycle recipient registry is full")
            if (
                current is None
                and recipients is not None
                and len(recipients) >= self._resource_lifecycle_recipient_limit
            ):
                raise RuntimeError("resource lifecycle recipient registry is full")
            if current is not None:
                current.close("replacement")
            recipients = self._resource_lifecycle_recipients.setdefault(
                operation, OrderedDict()
            )
            self._resource_lifecycle_recipient_version += 1
            token = _ResourceLifecycleRecipientToken(
                operation,
                name,
                self._resource_lifecycle_recipient_version,
                None if owner is None else id(owner),
            )
            bus_ref = ref(self)

            def retire(retired_token, owner_ref, reason):
                bus = bus_ref()
                if bus is not None and not sip.isdeleted(bus):
                    bus._retire_resource_lifecycle_recipient(
                        retired_token, owner_ref, reason
                    )

            recipients[name] = _GuardedResourceLifecycleRecipient(
                token, recipient, owner, retire
            )
            return token

    def unregister_resource_lifecycle_recipient(
        self, operation, name, recipient=None
    ) -> None:
        with self._resource_lifecycle_lock:
            recipients = self._resource_lifecycle_recipients.get(operation)
            if recipients is None:
                return
            current = recipients.get(name)
            if current is None or (
                recipient is not None
                and not current.matches_recipient(recipient)
            ):
                return
            current.close("explicit-unregister")

    def _retire_resource_lifecycle_recipient(
        self, token, owner_ref, reason
    ) -> None:
        with self._resource_lifecycle_lock:
            recipients = self._resource_lifecycle_recipients.get(
                token.operation
            )
            current = None if recipients is None else recipients.get(token.name)
            if current is not None and current.token is token:
                recipients.pop(token.name, None)
                if not recipients:
                    self._resource_lifecycle_recipients.pop(
                        token.operation, None
                    )
            retirements = self._resource_lifecycle_recipient_retirements
            retirements[token] = _ResourceLifecycleRecipientRetirement(
                token, owner_ref, reason
            )
            retirements.move_to_end(token)
            self._prune_resource_lifecycle_recipient_retirements()

    def _prune_resource_lifecycle_recipient_retirements(self) -> None:
        retirements = self._resource_lifecycle_recipient_retirements
        while len(retirements) > self._resource_lifecycle_history_limit:
            pending_tokens = {
                token
                for delivery in self._resource_lifecycle_pending.values()
                for token in delivery.recipients
            }
            removable = next(
                (token for token in retirements if token not in pending_tokens),
                None,
            )
            if removable is None:
                return
            retirements.pop(removable, None)

    @property
    def pending_resource_lifecycle_request_count(self) -> int:
        return len(self._resource_lifecycle_pending)

    @property
    def completed_resource_lifecycle_request_count(self) -> int:
        return len(self._resource_lifecycle_completed)

    @property
    def abandoned_resource_lifecycle_request_count(self) -> int:
        return len(self._resource_lifecycle_abandoned)

    @property
    def resource_lifecycle_pending_limit(self) -> int:
        return self._resource_lifecycle_pending_limit

    @property
    def resource_lifecycle_history_limit(self) -> int:
        return self._resource_lifecycle_history_limit

    def publish_resource_lifecycle(
        self, request: ResourceLifecycleRequested
    ) -> bool:
        """Deliver once per exact recipient, retaining only retryable NACKs."""
        with self._resource_lifecycle_lock:
            if (
                not self._resource_lifecycle_dispatch_active
                or type(request) is not ResourceLifecycleRequested
            ):
                return False
            key = (request.shutdown_generation, request.operation)
            completion = self._resource_lifecycle_completed.get(key)
            if completion is not None:
                return completion.request is request
            delivery = self._resource_lifecycle_pending.get(key)
            if delivery is None:
                recipients = self._resource_lifecycle_recipients.get(
                    request.operation
                )
                if not recipients:
                    return False
                if (
                    len(self._resource_lifecycle_pending)
                    >= self._resource_lifecycle_pending_limit
                ):
                    return False
                delivery = _ResourceLifecycleDelivery(
                    request,
                    tuple(
                        recipient.token for recipient in recipients.values()
                    ),
                )
                self._resource_lifecycle_pending[key] = delivery
            elif delivery.request is not request:
                return False
            if delivery.in_progress:
                return False
            delivery.in_progress = True
        try:
            for token in delivery.recipients:
                with self._resource_lifecycle_lock:
                    if (
                        not self._resource_lifecycle_dispatch_active
                        or self._resource_lifecycle_pending.get(key)
                        is not delivery
                    ):
                        return False
                    if token in delivery.acknowledged:
                        continue
                    recipients = self._resource_lifecycle_recipients.get(
                        request.operation, {}
                    )
                    recipient = recipients.get(token.name)
                    if recipient is None or recipient.token is not token:
                        continue
                result = recipient.deliver(request)
                with self._resource_lifecycle_lock:
                    if (
                        not self._resource_lifecycle_dispatch_active
                        or self._resource_lifecycle_pending.get(key)
                        is not delivery
                    ):
                        return False
                    current = self._resource_lifecycle_recipients.get(
                        request.operation, {}
                    ).get(token.name)
                    if current is not recipient or current.token is not token:
                        continue
                    if result is ResourceLifecycleRecipientResult.ACK:
                        delivery.acknowledged.add(token)
        finally:
            with self._resource_lifecycle_lock:
                if self._resource_lifecycle_pending.get(key) is delivery:
                    delivery.in_progress = False
        with self._resource_lifecycle_lock:
            if (
                not self._resource_lifecycle_dispatch_active
                or self._resource_lifecycle_pending.get(key) is not delivery
                or len(delivery.acknowledged) != len(delivery.recipients)
            ):
                return False
            self._complete_resource_lifecycle_request(key)
            return True

    def acknowledge_resource_lifecycle_recipient(
        self, request, recipient_token
    ) -> bool:
        """Resolve an exact token only after its QObject is provably retired."""
        if (
            type(request) is not ResourceLifecycleRequested
            or type(recipient_token) is not _ResourceLifecycleRecipientToken
            or recipient_token.owner_identity is None
        ):
            return False
        return self.resolve_retired_resource_lifecycle_recipient(
            request,
            recipient_token,
            owner_identity=recipient_token.owner_identity,
            registration_generation=recipient_token.version,
        )

    def resolve_retired_resource_lifecycle_recipient(
        self,
        request,
        recipient_token,
        *,
        owner_identity,
        registration_generation,
    ) -> bool:
        """Resolve only an exact token whose registered QObject is provably dead."""
        if (
            type(request) is not ResourceLifecycleRequested
            or type(recipient_token) is not _ResourceLifecycleRecipientToken
            or type(owner_identity) is not int
            or type(registration_generation) is not int
        ):
            return False
        with self._resource_lifecycle_lock:
            retirement = self._resource_lifecycle_recipient_retirements.get(
                recipient_token
            )
            if (
                retirement is None
                or retirement.token is not recipient_token
                or recipient_token.owner_identity != owner_identity
                or recipient_token.version != registration_generation
                or retirement.owner_ref is None
            ):
                return False
            try:
                owner = retirement.owner_ref()
                owner_is_dead = owner is None or (
                    isinstance(owner, QObject) and sip.isdeleted(owner)
                )
            except BaseException:
                return False
            if not owner_is_dead:
                return False
            return self._acknowledge_resource_lifecycle_recipient_locked(
                request, recipient_token
            )

    def _acknowledge_resource_lifecycle_recipient_locked(
        self, request, recipient_token
    ) -> bool:
        key = (request.shutdown_generation, request.operation)
        delivery = self._resource_lifecycle_pending.get(key)
        if (
            not self._resource_lifecycle_dispatch_active
            or delivery is None
            or delivery.request is not request
            or not any(token is recipient_token for token in delivery.recipients)
        ):
            return False
        delivery.acknowledged.add(recipient_token)
        if len(delivery.acknowledged) == len(delivery.recipients):
            self._complete_resource_lifecycle_request(key)
        return True

    def is_resource_lifecycle_request_completed(self, request) -> bool:
        if type(request) is not ResourceLifecycleRequested:
            return False
        key = (request.shutdown_generation, request.operation)
        with self._resource_lifecycle_lock:
            completion = self._resource_lifecycle_completed.get(key)
            return completion is not None and completion.request is request

    def _complete_resource_lifecycle_request(self, key) -> None:
        delivery = self._resource_lifecycle_pending.pop(key, None)
        if delivery is None:
            return
        completed = self._resource_lifecycle_completed
        completed[key] = _ResourceLifecycleCompletion(delivery.request)
        completed.move_to_end(key)
        while len(completed) > self._resource_lifecycle_history_limit:
            completed.popitem(last=False)

    def close_resource_lifecycle_dispatcher(self, *_args) -> None:
        with self._resource_lifecycle_lock:
            if not self._resource_lifecycle_dispatch_active:
                return
            self._resource_lifecycle_dispatch_active = False
            abandoned = self._resource_lifecycle_abandoned
            for key, delivery in tuple(
                self._resource_lifecycle_pending.items()
            ):
                abandoned[key] = _ResourceLifecycleAbandonment(
                    delivery.request.shutdown_generation,
                    delivery.request.operation,
                    delivery.recipients,
                    "event-bus-disconnect",
                )
                abandoned.move_to_end(key)
                while len(abandoned) > self._resource_lifecycle_history_limit:
                    abandoned.popitem(last=False)
            self._resource_lifecycle_pending.clear()
            for recipients in tuple(
                self._resource_lifecycle_recipients.values()
            ):
                for recipient in tuple(recipients.values()):
                    recipient.close("dispatcher-close")
            self._resource_lifecycle_recipients.clear()
            self._resource_lifecycle_recipient_retirements.clear()

    def register_workflow_continuation_lifecycle_owner(self, owner) -> None:
        """Authorize one exact QObject to abandon only its own deliveries."""
        if not isinstance(owner, QObject) or sip.isdeleted(owner):
            raise TypeError("continuation lifecycle owner must be a QObject")
        self._prune_workflow_continuation_lifecycle_owners()
        owner_key = id(owner)
        current = self._workflow_continuation_lifecycle_owners.get(owner_key)
        if current is not None and current[0]() is owner:
            return
        if (
            len(self._workflow_continuation_lifecycle_owners)
            >= self._workflow_continuation_lifecycle_owner_limit
        ):
            raise RuntimeError(
                "workflow continuation lifecycle owner registry is full"
            )
        self._workflow_continuation_lifecycle_owner_version += 1
        token = _ContinuationLifecycleOwnerToken(
            self._workflow_continuation_lifecycle_owner_version
        )
        owner_ref = ref(owner)
        self._workflow_continuation_lifecycle_owners[owner_key] = (
            owner_ref,
            token,
        )

    def unregister_workflow_continuation_lifecycle_owner(self, owner) -> None:
        current = self._workflow_continuation_lifecycle_owners.get(id(owner))
        if current is not None and current[0]() is owner:
            self._workflow_continuation_lifecycle_owners.pop(id(owner), None)

    def _prune_workflow_continuation_lifecycle_owners(self) -> None:
        for owner_key, (owner_ref, _token) in tuple(
            self._workflow_continuation_lifecycle_owners.items()
        ):
            owner = owner_ref()
            if owner is None or sip.isdeleted(owner):
                self._workflow_continuation_lifecycle_owners.pop(
                    owner_key, None
                )

    def _workflow_continuation_lifecycle_owner_token(self, owner):
        current = self._workflow_continuation_lifecycle_owners.get(id(owner))
        if (
            current is None
            or current[0]() is not owner
            or sip.isdeleted(owner)
        ):
            if current is not None:
                self._workflow_continuation_lifecycle_owners.pop(
                    id(owner), None
                )
            return None
        return current[1]

    def register_workflow_continuation_recipient(
        self, kind, name, recipient, *, owner=None
    ):
        """Register a recipient and return its opaque exact-delivery token."""
        if type(kind) is not str or not kind:
            raise ValueError("continuation kind must be a non-empty plain string")
        if type(name) is not str or not name or not callable(recipient):
            raise ValueError("continuation recipient requires a name and callback")
        if owner is not None and not isinstance(owner, QObject):
            raise TypeError("continuation recipient owner must be a QObject")
        recipients = self._workflow_continuation_recipients.get(kind)
        current = None if recipients is None else recipients.get(name)
        if current is not None and current.matches_registration(recipient, owner):
            return current.token
        if current is None and (
            self.workflow_continuation_recipient_count
            >= self._workflow_continuation_recipient_total_limit
        ):
            raise RuntimeError("workflow continuation recipient registry is full")
        if recipients is not None and (
            current is None
            and len(recipients) >= self._workflow_continuation_recipient_limit
        ):
            raise RuntimeError("workflow continuation recipient registry is full")
        if current is not None:
            current.close()
        recipients = self._workflow_continuation_recipients.setdefault(
            kind, OrderedDict()
        )
        self._workflow_continuation_recipient_version += 1
        token = _ContinuationRecipientToken(
            kind,
            name,
            self._workflow_continuation_recipient_version,
        )
        bus_ref = ref(self)

        def retire(retired_token):
            bus = bus_ref()
            if bus is not None and not sip.isdeleted(bus):
                bus._retire_workflow_continuation_recipient(retired_token)

        recipients[name] = _GuardedContinuationRecipient(
            token, recipient, owner, retire
        )
        return token

    def unregister_workflow_continuation_recipient(
        self, kind, name, recipient=None
    ) -> None:
        recipients = self._workflow_continuation_recipients.get(kind)
        if recipients is None:
            return
        current = recipients.get(name)
        if current is None or (
            recipient is not None and not current.matches_recipient(recipient)
        ):
            return
        current.close()

    def _retire_workflow_continuation_recipient(self, token) -> None:
        recipients = self._workflow_continuation_recipients.get(token.kind)
        current = None if recipients is None else recipients.get(token.name)
        if current is not None and current.token is token:
            recipients.pop(token.name, None)
            if not recipients:
                self._workflow_continuation_recipients.pop(token.kind, None)

    @property
    def pending_workflow_continuation_delivery_count(self) -> int:
        return len(self._workflow_continuation_pending)

    @property
    def workflow_continuation_recipient_limit(self) -> int:
        return self._workflow_continuation_recipient_limit

    @property
    def workflow_continuation_lifecycle_owner_limit(self) -> int:
        return self._workflow_continuation_lifecycle_owner_limit

    @property
    def workflow_continuation_lifecycle_owner_count(self) -> int:
        self._prune_workflow_continuation_lifecycle_owners()
        return len(self._workflow_continuation_lifecycle_owners)

    @property
    def workflow_continuation_recipient_total_limit(self) -> int:
        return self._workflow_continuation_recipient_total_limit

    @property
    def workflow_continuation_recipient_count(self) -> int:
        return sum(
            len(recipients)
            for recipients in self._workflow_continuation_recipients.values()
        )

    @property
    def workflow_continuation_kind_count(self) -> int:
        return len(self._workflow_continuation_recipients)

    @property
    def completed_workflow_continuation_delivery_count(self) -> int:
        return len(self._workflow_continuation_completed)

    @property
    def abandoned_workflow_continuation_delivery_count(self) -> int:
        return len(self._workflow_continuation_abandoned)

    @property
    def abandoned_workflow_continuation_diagnostics(self) -> tuple:
        return tuple(self._workflow_continuation_abandoned.values())

    def _workflow_continuation_completed_for_other_owner(
        self, stable_delivery_id, lifecycle_owner_token
    ) -> bool:
        return any(
            completed.delivery_id == stable_delivery_id
            and completed.lifecycle_owner_token != lifecycle_owner_token
            for completed in self._workflow_continuation_completed.values()
        )

    def deliver_workflow_continuation(
        self, delivery_id, kind, message, *, owner=None
    ) -> bool:
        return bool(
            self.deliver_workflow_continuation_outcome(
                delivery_id, kind, message, owner=owner
            ).status
            is WorkflowContinuationDeliveryStatus.ACK
        )

    def deliver_workflow_continuation_outcome(
        self, delivery_id, kind, message, *, owner=None
    ) -> WorkflowContinuationDeliveryOutcome:
        """Deliver directly with exact publisher and recipient outcomes."""
        outcome_type = WorkflowContinuationDeliveryOutcome
        ack = WorkflowContinuationDeliveryStatus.ACK
        retryable = WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
        permanent = WorkflowContinuationDeliveryStatus.PERMANENT_REJECT

        def outcome(status, reason=""):
            return outcome_type(status, reason[:256])

        if not self._workflow_continuation_dispatch_active:
            return outcome(permanent, "continuation dispatcher is inactive")
        lifecycle_owner_token = (
            self._workflow_continuation_lifecycle_owner_token(owner)
        )
        if lifecycle_owner_token is None:
            return outcome(permanent, "continuation lifecycle owner is unauthorized")
        if type(kind) is not str or kind not in _CONTINUATION_MESSAGE_CONTRACTS:
            return outcome(permanent, "continuation kind is unsupported")
        stable_delivery_id = _continuation_delivery_id(delivery_id, kind)
        if stable_delivery_id is None:
            return outcome(permanent, "continuation delivery identifier is invalid")
        message_identity, identity_reason = (
            _continuation_message_identity_outcome(kind, message)
        )
        if message_identity is None:
            return outcome(permanent, identity_reason)
        composite_key = (lifecycle_owner_token, stable_delivery_id)
        completion = self._workflow_continuation_completed.get(composite_key)
        if completion is not None:
            if (
                completion.kind == kind
                and completion.message_identity == message_identity
            ):
                return outcome(ack)
            return outcome(permanent, "completed continuation identity conflicts")
        if composite_key in self._workflow_continuation_abandoned:
            return outcome(permanent, "continuation delivery was abandoned")
        pending = self._workflow_continuation_pending
        delivery = pending.get(composite_key)
        if self._workflow_continuation_completed_for_other_owner(
            stable_delivery_id, lifecycle_owner_token
        ):
            reason = "continuation delivery identifier belongs to another owner"
            if delivery is not None:
                self._abandon_workflow_continuation_ids(
                    (composite_key,),
                    reason=reason,
                    lifecycle_owner_token=lifecycle_owner_token,
                )
            return outcome(permanent, reason)
        if delivery is not None and (
            delivery.kind != kind
            or delivery.message_identity != message_identity
        ):
            reason = "pending continuation identity conflicts"
            self._abandon_workflow_continuation_ids(
                (composite_key,),
                reason=reason,
                lifecycle_owner_token=lifecycle_owner_token,
            )
            return outcome(permanent, reason)
        if composite_key in self._workflow_continuation_active:
            return outcome(retryable, "continuation delivery is already active")
        if delivery is None:
            recipients = self._workflow_continuation_recipients.get(kind)
            if not recipients:
                return outcome(retryable, "continuation has no registered recipients")
            if len(pending) >= self._workflow_continuation_history_limit:
                return outcome(retryable, "continuation pending capacity is exhausted")
            delivery = _WorkflowContinuationDelivery(
                kind=kind,
                message=message,
                message_identity=message_identity,
                recipients=tuple(
                    recipient.token for recipient in recipients.values()
                ),
                lifecycle_owner_token=lifecycle_owner_token,
            )
            pending[composite_key] = delivery

        recipients = self._workflow_continuation_recipients.get(kind, {})
        for token in delivery.recipients:
            if any(
                acknowledged is token
                for acknowledged in delivery.acknowledged
            ):
                continue
            recipient = recipients.get(token.name)
            if (
                recipient is not None
                and recipient.token is token
            ):
                self._workflow_continuation_active.add(composite_key)
                try:
                    recipient_outcome = recipient.deliver(message)
                finally:
                    self._workflow_continuation_active.discard(composite_key)
                if pending.get(composite_key) is not delivery:
                    abandonment = self._workflow_continuation_abandoned.get(
                        composite_key
                    )
                    reason = (
                        abandonment.reason
                        if abandonment is not None
                        else "continuation delivery changed during recipient callback"
                    )
                    return outcome(permanent, reason)
                if self._workflow_continuation_completed_for_other_owner(
                    stable_delivery_id, lifecycle_owner_token
                ):
                    reason = (
                        "continuation delivery identifier belongs to another owner"
                    )
                    self._abandon_workflow_continuation_ids(
                        (composite_key,),
                        reason=reason,
                        lifecycle_owner_token=lifecycle_owner_token,
                    )
                    return outcome(permanent, reason)
                if recipient_outcome is WorkflowContinuationRecipientResult.ACK:
                    delivery.acknowledged.add(token)
                elif recipient_outcome is (
                    WorkflowContinuationRecipientResult.PERMANENT_REJECT
                ):
                    reason = "continuation recipient permanently rejected"
                    self._abandon_workflow_continuation_ids(
                        (composite_key,),
                        reason=reason,
                        lifecycle_owner_token=lifecycle_owner_token,
                    )
                    return outcome(permanent, reason)

        if pending.get(composite_key) is not delivery:
            return outcome(
                permanent,
                "continuation delivery changed before completion",
            )
        if len(delivery.acknowledged) != len(delivery.recipients):
            return outcome(retryable, "continuation recipient acknowledgement pending")
        self._complete_workflow_continuation(composite_key)
        return outcome(ack)

    def acknowledge_workflow_continuation_recipient(
        self,
        delivery_id,
        kind,
        message,
        recipient_token,
    ) -> bool:
        """Terminally ACK one exact pending recipient after its owner retires it."""
        if type(recipient_token) is not _ContinuationRecipientToken:
            return False
        if recipient_token.kind != kind:
            return False
        stable_delivery_id = _continuation_delivery_id(delivery_id, kind)
        if stable_delivery_id is None:
            return False
        message_identity = _continuation_message_identity(kind, message)
        if message_identity is None:
            return False
        matches = [
            (composite_key, delivery)
            for composite_key, delivery in (
                self._workflow_continuation_pending.items()
            )
            if composite_key[1] == stable_delivery_id
            and delivery.kind == kind
            and delivery.message is message
            and delivery.message_identity == message_identity
            and any(
                token is recipient_token for token in delivery.recipients
            )
        ]
        if len(matches) != 1:
            return False
        composite_key, delivery = matches[0]
        if composite_key in self._workflow_continuation_active:
            return False
        lifecycle_owner_token, stable_delivery_id = composite_key
        if self._workflow_continuation_completed_for_other_owner(
            stable_delivery_id, lifecycle_owner_token
        ):
            self._abandon_workflow_continuation_ids(
                (composite_key,),
                reason=(
                    "continuation delivery identifier belongs to another owner"
                ),
                lifecycle_owner_token=lifecycle_owner_token,
            )
            return False
        if self._workflow_continuation_pending.get(composite_key) is not delivery:
            return False
        delivery.acknowledged.add(recipient_token)
        if len(delivery.acknowledged) == len(delivery.recipients):
            if (
                self._workflow_continuation_pending.get(composite_key)
                is not delivery
            ):
                return False
            self._complete_workflow_continuation(composite_key)
        return True

    def _complete_workflow_continuation(self, composite_key) -> None:
        pending = self._workflow_continuation_pending
        delivery = pending.pop(composite_key, None)
        if delivery is None:
            return
        lifecycle_owner_token, delivery_id = composite_key
        completed = self._workflow_continuation_completed
        completed[composite_key] = _WorkflowContinuationCompletion(
            lifecycle_owner_token,
            delivery_id,
            delivery.kind,
            delivery.message_identity,
        )
        completed.move_to_end(composite_key)
        while len(completed) > self._workflow_continuation_history_limit:
            completed.popitem(last=False)

    def abandon_workflow_continuations(
        self, delivery_ids, *, owner, reason
    ) -> int:
        """Explicitly abandon exact deliveries owned by one lifecycle owner."""
        if type(delivery_ids) is not tuple:
            raise TypeError("continuation delivery ids must be an exact tuple")
        if type(reason) is not str or not reason:
            raise ValueError("continuation abandonment requires a reason")
        lifecycle_owner_token = (
            self._workflow_continuation_lifecycle_owner_token(owner)
        )
        if lifecycle_owner_token is None:
            return 0
        stable_delivery_ids = []
        for delivery_id in tuple.__iter__(delivery_ids):
            if type(delivery_id) is not tuple or not delivery_id:
                raise ValueError("continuation delivery id must be a tuple")
            kind = tuple.__getitem__(delivery_id, 0)
            stable_delivery_id = _continuation_delivery_id(
                delivery_id, kind
            )
            if stable_delivery_id is None:
                raise ValueError("continuation delivery id is unsupported")
            stable_delivery_ids.append(stable_delivery_id)
        return self._abandon_workflow_continuation_ids(
            tuple(
                (lifecycle_owner_token, delivery_id)
                for delivery_id in stable_delivery_ids
            ),
            reason=reason,
            lifecycle_owner_token=lifecycle_owner_token,
        )

    def _abandon_workflow_continuation_ids(
        self,
        delivery_ids,
        *,
        reason,
        lifecycle_owner_token=None,
        require_owner=True,
    ) -> int:
        abandoned_count = 0
        pending = self._workflow_continuation_pending
        abandoned = self._workflow_continuation_abandoned
        for composite_key in delivery_ids:
            delivery = pending.get(composite_key)
            if delivery is None or (
                require_owner
                and delivery.lifecycle_owner_token != lifecycle_owner_token
            ):
                continue
            pending.pop(composite_key, None)
            _owner_token, delivery_id = composite_key
            abandoned[composite_key] = _WorkflowContinuationAbandonment(
                delivery_id,
                delivery.kind,
                delivery.recipients,
                delivery.lifecycle_owner_token,
                reason[:256],
            )
            abandoned.move_to_end(composite_key)
            while len(abandoned) > self._workflow_continuation_history_limit:
                abandoned.popitem(last=False)
            abandoned_count += 1
        return abandoned_count

    def close_workflow_continuation_dispatcher(self, *_args) -> None:
        if not self._workflow_continuation_dispatch_active:
            return
        self._workflow_continuation_dispatch_active = False
        self._abandon_workflow_continuation_ids(
            tuple(self._workflow_continuation_pending),
            reason="event-bus-disconnect",
            require_owner=False,
        )
        for recipients in tuple(self._workflow_continuation_recipients.values()):
            for recipient in tuple(recipients.values()):
                recipient.close()
        self._workflow_continuation_recipients.clear()
        self._workflow_continuation_lifecycle_owners.clear()

    def register_export_retry_recipient(self, recipient) -> None:
        """Install the canonical synchronous Workflow retry-ack port."""
        self._export_retry_recipient = recipient

    def unregister_export_retry_recipient(self, recipient) -> None:
        if self._export_retry_recipient == recipient:
            self._export_retry_recipient = None

    def deliver_export_retry_accepted(self, message) -> bool:
        recipient = self._export_retry_recipient
        if recipient is None:
            return False
        try:
            return recipient(message) is True
        except BaseException:
            return False

    def register_export_terminal_recipient(
        self, name, recipient, *, critical=False
    ) -> None:
        if type(name) is not str or not name or not callable(recipient):
            raise ValueError("terminal recipient requires a name and callback")
        self._export_terminal_recipients[name] = (
            recipient,
            bool(critical),
        )

    def register_import_terminal_recipient(
        self, name, recipient, *, owner=None, critical=False
    ) -> None:
        if type(name) is not str or not name or not callable(recipient):
            raise ValueError("terminal recipient requires a name and callback")
        if owner is None:
            bound_owner = getattr(recipient, "__self__", None)
            if isinstance(bound_owner, QObject):
                owner = bound_owner
        if owner is not None and (
            not isinstance(owner, QObject) or sip.isdeleted(owner)
        ):
            raise TypeError("import terminal recipient owner must be a live QObject")
        current = self._import_terminal_recipients.get(name)
        if (
            current is not None
            and current.token.critical is bool(critical)
            and current.matches_registration(recipient, owner)
        ):
            return
        if (
            current is None
            and len(self._import_terminal_recipients)
            >= self._import_terminal_recipient_limit
        ):
            raise RuntimeError("import terminal recipient registry is full")
        if current is not None:
            current.close()
        self._import_terminal_recipient_version += 1
        token = _ImportTerminalRecipientToken(
            name, self._import_terminal_recipient_version, bool(critical)
        )
        bus_ref = ref(self)

        def retire(retired_token):
            bus = bus_ref()
            if bus is not None and not sip.isdeleted(bus):
                bus._retire_import_terminal_recipient(retired_token)

        self._import_terminal_recipients[name] = _GuardedImportTerminalRecipient(
            token, recipient, owner, retire
        )

    def unregister_import_terminal_recipient(self, name, recipient=None) -> None:
        current = self._import_terminal_recipients.get(name)
        if current is None or (
            recipient is not None and not current.matches_recipient(recipient)
        ):
            return
        current.close()

    def _retire_import_terminal_recipient(self, token) -> None:
        current = self._import_terminal_recipients.get(token.name)
        if current is not None and current.token == token:
            self._import_terminal_recipients.pop(token.name, None)

    def has_import_terminal_recipients(self) -> bool:
        return bool(self._import_terminal_recipients)

    def has_critical_import_terminal_recipient(self) -> bool:
        return any(
            recipient.token.critical
            for recipient in self._import_terminal_recipients.values()
        )

    @property
    def import_terminal_recipient_limit(self) -> int:
        return self._import_terminal_recipient_limit

    @property
    def import_terminal_recipient_count(self) -> int:
        return len(self._import_terminal_recipients)

    @property
    def import_terminal_pending_limit(self) -> int:
        return self._import_terminal_pending_limit

    @property
    def pending_import_terminal_delivery_count(self) -> int:
        return len(self._import_terminal_pending)

    @property
    def waiting_import_terminal_delivery_count(self) -> int:
        return int(self._import_terminal_capacity_waiting is not None)

    @property
    def import_terminal_claim_count(self) -> int:
        return len(self._import_terminal_claims)

    @property
    def import_terminal_claim_limit(self) -> int:
        return (
            self._import_terminal_pending_limit
            + (2 * self._import_terminal_history_limit)
            + 1
        )

    @property
    def completed_import_terminal_delivery_count(self) -> int:
        return len(self._import_terminal_completed)

    @property
    def abandoned_import_terminal_delivery_count(self) -> int:
        return len(self._import_terminal_abandoned)

    @property
    def import_terminal_history_limit(self) -> int:
        return self._import_terminal_history_limit

    @staticmethod
    def _import_terminal_identity(delivery_id, message):
        if type(message) not in {ImportedAudioReady, ImportedAudioFailed}:
            return None
        if (
            type(delivery_id) is not tuple
            or tuple.__len__(delivery_id) != 2
            or type(tuple.__getitem__(delivery_id, 0)) is not str
            or type(tuple.__getitem__(delivery_id, 1)) is not str
        ):
            return None
        expected = (type(message).__name__, message.import_id)
        return expected if delivery_id == expected else None

    def deliver_import_terminal(self, delivery_id, message) -> bool:
        """Deliver an import terminal once, with critical Workflow first."""
        if not self._import_terminal_dispatch_active:
            return False
        stable_delivery_id = self._import_terminal_identity(delivery_id, message)
        if stable_delivery_id is None:
            return False
        completion = self._import_terminal_completed.get(stable_delivery_id)
        if completion is not None:
            return completion.message is message
        if stable_delivery_id in self._import_terminal_abandoned:
            return False
        import_id = message.import_id
        claim = self._import_terminal_claims.get(import_id)
        if claim is not None and (
            claim[0] != stable_delivery_id or claim[1] is not message
        ):
            return False
        delivery = self._import_terminal_pending.get(stable_delivery_id)
        if delivery is None:
            if not self.has_critical_import_terminal_recipient():
                return False
            if claim is None:
                # Preserve one exact first outcome before checking active
                # capacity. Sequence production admits only one active import,
                # so one reservation protects the legitimate overflow without
                # permitting an unbounded claim map.
                if self._import_terminal_capacity_waiting is not None:
                    return False
                waiting = _ImportTerminalCompletion(
                    stable_delivery_id, message
                )
                self._import_terminal_capacity_waiting = waiting
                self._import_terminal_claims[import_id] = (
                    stable_delivery_id,
                    message,
                )
            if (
                len(self._import_terminal_pending)
                >= self._import_terminal_pending_limit
            ):
                return False
            waiting = self._import_terminal_capacity_waiting
            if (
                waiting is None
                or waiting.delivery_id != stable_delivery_id
                or waiting.message is not message
            ):
                return False
            recipients = tuple(
                sorted(
                    (
                        recipient.token
                        for recipient in self._import_terminal_recipients.values()
                    ),
                    key=lambda token: not token.critical,
                )
            )
            delivery = _ImportTerminalDelivery(
                stable_delivery_id, message, recipients
            )
            self._import_terminal_capacity_waiting = None
            self._import_terminal_pending[stable_delivery_id] = delivery
        elif delivery.message is not message:
            return False

        if delivery.in_progress:
            return False

        recipients = self._import_terminal_recipients
        delivery.in_progress = True
        try:
            for token in delivery.recipients:
                if token in delivery.acknowledged:
                    continue
                recipient = recipients.get(token.name)
                matches = (
                    recipient is not None
                    and recipient.token == token
                )
                result = (
                    recipient.deliver(delivery.message)
                    if matches
                    else ImportTerminalRecipientResult.RETRYABLE_NACK
                )
                if token.critical:
                    if result is ImportTerminalRecipientResult.PERMANENT_REJECT:
                        self.abandon_import_terminal(
                            stable_delivery_id, "recipient-permanent-reject"
                        )
                        return False
                    if result is not ImportTerminalRecipientResult.ACK:
                        return False
                    delivery.acknowledged.add(token)
                elif matches:
                    # Compatibility observers cannot veto or force retry of the
                    # already-acknowledged Workflow business outcome.
                    delivery.acknowledged.add(token)
            required = tuple(
                token for token in delivery.recipients if token.critical
            )
            if any(token not in delivery.acknowledged for token in required):
                return False
            self._import_terminal_pending.pop(stable_delivery_id, None)
            self._import_terminal_completed[stable_delivery_id] = (
                _ImportTerminalCompletion(stable_delivery_id, delivery.message)
            )
            self._import_terminal_completed.move_to_end(stable_delivery_id)
            if (
                len(self._import_terminal_completed)
                > self._import_terminal_history_limit
            ):
                old_completion = self._import_terminal_completed.popitem(
                    last=False
                )[1]
                self._release_import_terminal_claim(old_completion)
            return True
        finally:
            delivery.in_progress = False

    def _release_import_terminal_claim(self, terminal) -> None:
        claim = self._import_terminal_claims.get(terminal.message.import_id)
        if (
            claim is not None
            and claim[0] == terminal.delivery_id
            and claim[1] is terminal.message
        ):
            self._import_terminal_claims.pop(terminal.message.import_id, None)

    def abandon_import_terminal(self, delivery_id, reason) -> bool:
        waiting = self._import_terminal_capacity_waiting
        is_waiting = (
            waiting is not None and waiting.delivery_id == delivery_id
        )
        if (
            delivery_id in self._import_terminal_completed
            or delivery_id in self._import_terminal_abandoned
            or (
                delivery_id not in self._import_terminal_pending
                and not is_waiting
            )
        ):
            return False
        self._import_terminal_pending.pop(delivery_id, None)
        if is_waiting:
            self._import_terminal_capacity_waiting = None
        self._import_terminal_abandoned[delivery_id] = str(
            reason or "import-terminal-abandoned"
        )[:256]
        self._import_terminal_abandoned.move_to_end(delivery_id)
        if len(self._import_terminal_abandoned) > self._import_terminal_history_limit:
            old_delivery_id, _reason = self._import_terminal_abandoned.popitem(
                last=False
            )
            old_claim = next(
                (
                    claim
                    for claim in self._import_terminal_claims.values()
                    if claim[0] == old_delivery_id
                ),
                None,
            )
            if old_claim is not None:
                self._import_terminal_claims.pop(old_claim[1].import_id, None)
        return True

    def import_terminal_abandonment_reason(self, delivery_id):
        return self._import_terminal_abandoned.get(delivery_id)

    def classify_import_terminal_delivery(
        self, delivery_id, message
    ) -> ImportTerminalRecipientResult:
        """Report whether one exact terminal may be retried or is resolved."""
        stable_delivery_id = self._import_terminal_identity(
            delivery_id, message
        )
        if (
            stable_delivery_id is None
            or not self._import_terminal_dispatch_active
        ):
            return ImportTerminalRecipientResult.PERMANENT_REJECT
        completion = self._import_terminal_completed.get(stable_delivery_id)
        if completion is not None:
            return (
                ImportTerminalRecipientResult.ACK
                if completion.message is message
                else ImportTerminalRecipientResult.PERMANENT_REJECT
            )
        if stable_delivery_id in self._import_terminal_abandoned:
            return ImportTerminalRecipientResult.PERMANENT_REJECT
        claim = self._import_terminal_claims.get(message.import_id)
        if claim is None:
            return ImportTerminalRecipientResult.RETRYABLE_NACK
        if claim[0] == stable_delivery_id and claim[1] is message:
            return ImportTerminalRecipientResult.RETRYABLE_NACK
        return ImportTerminalRecipientResult.PERMANENT_REJECT

    def close_import_terminal_dispatcher(self, *_args) -> None:
        if not self._import_terminal_dispatch_active:
            return
        self._import_terminal_dispatch_active = False
        for delivery_id in tuple(self._import_terminal_pending):
            self.abandon_import_terminal(delivery_id, "event-bus-disconnect")
        waiting = self._import_terminal_capacity_waiting
        if waiting is not None:
            self.abandon_import_terminal(
                waiting.delivery_id, "event-bus-disconnect"
            )
        for recipient in tuple(self._import_terminal_recipients.values()):
            recipient.close()
        self._import_terminal_recipients.clear()

    def unregister_export_terminal_recipient(self, name, recipient=None) -> None:
        current = self._export_terminal_recipients.get(name)
        if current is None:
            return
        if recipient is None or current[0] == recipient:
            self._export_terminal_recipients.pop(name, None)

    def has_export_terminal_recipients(self) -> bool:
        return bool(self._export_terminal_recipients)

    def deliver_export_terminal(self, delivery_id, message) -> bool:
        """Deliver once per recipient and retain only unfinished acknowledgements."""
        if delivery_id in self._export_terminal_completed:
            return True
        delivered = self._export_terminal_pending.setdefault(
            delivery_id, set()
        )
        recipients = tuple(
            sorted(
                self._export_terminal_recipients.items(),
                key=lambda item: not item[1][1],
            )
        )
        for name, (recipient, critical) in recipients:
            if name in delivered:
                continue
            try:
                accepted = recipient(message) is True
            except BaseException:
                if critical:
                    return False
                delivered.add(name)
                continue
            if not accepted and critical:
                return False
            delivered.add(name)
        self._export_terminal_pending.pop(delivery_id, None)
        self._export_terminal_completed[delivery_id] = None
        self._export_terminal_completed.move_to_end(delivery_id)
        if len(self._export_terminal_completed) > self._export_terminal_history_limit:
            self._export_terminal_completed.popitem(last=False)
        return True
