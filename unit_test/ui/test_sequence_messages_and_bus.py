import ast
import copy
import ctypes
import gc
import hashlib
import inspect
import os
import pickle
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from enum import Enum, EnumType, IntEnum, StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Barrier, Thread
from types import MappingProxyType
from weakref import ref

import numpy as np
import pytest

import ui.sequence.sequence_event_bus as event_bus_module
import ui.sequence.sequence_workflow_controller as workflow_controller_module
from ui.sequence import sequence_messages as messages
from ui.sequence.sequence_event_bus import (
    ImportTerminalRecipientResult,
    RetainedCleanupLifecycleRegistrationResult,
    ResourceLifecycleRecipientResult,
    SequenceEventBus,
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
    WorkflowContinuationRecipientResult,
)
from ui.sequence.sequence_messages import (
    AnalysisTransportReady,
    AudioBatch,
    AudioCancelled,
    AudioCompleted,
    AudioFailed,
    BeginRecordingRequested,
    ConfigurationChanged,
    ConfigurationSnapshot,
    ExportCompleted,
    LoadImportedAudioRequested,
    RecordingBatchReady,
    RecordingCompleted,
    ResourceLifecycleRequested,
    RetryExportRequested,
    ShutdownFlushFailed,
    StartTestRequested,
)


def test_resource_lifecycle_request_is_exact_immutable_identity():
    request = ResourceLifecycleRequested(7, "disconnect-domains")

    assert request.shutdown_generation == 7
    assert request.operation == "disconnect-domains"
    with pytest.raises(FrozenInstanceError):
        request.operation = "disconnect-coordinator"
    with pytest.raises(ValueError):
        ResourceLifecycleRequested(-1, "disconnect-domains")
    with pytest.raises(ValueError):
        ResourceLifecycleRequested(7, "")


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("runtime"), KeyboardInterrupt(), SystemExit()],
)
def test_resource_lifecycle_dispatch_contains_failure_and_retries_only_nack(
    failure,
):
    bus = SequenceEventBus()
    calls = []
    accepted = {"value": False}

    def first(request):
        calls.append(("first", request))
        return ResourceLifecycleRecipientResult.ACK

    def second(request):
        calls.append(("second", request))
        if not accepted["value"]:
            if isinstance(failure, BaseException):
                raise failure
            return failure
        return True

    bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "first", first
    )
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "second", second
    )
    request = ResourceLifecycleRequested(11, "disconnect-domains")

    assert bus.publish_resource_lifecycle(request) is False
    assert calls == [("first", request), ("second", request)]
    assert bus.pending_resource_lifecycle_request_count == 1

    accepted["value"] = True
    assert bus.publish_resource_lifecycle(request) is True
    assert calls == [
        ("first", request),
        ("second", request),
        ("second", request),
    ]
    assert bus.publish_resource_lifecycle(request) is True
    assert len(calls) == 3
    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.completed_resource_lifecycle_request_count == 1


def test_resource_lifecycle_pending_keeps_exact_replaced_recipient_token(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    calls = []

    class Recipient(QObject):
        def receive(self, request):
            calls.append(("old", request))
            return False

    old = Recipient()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "workflow", old.receive, owner=old
    )
    request = ResourceLifecycleRequested(13, "disconnect-domains")
    assert bus.publish_resource_lifecycle(request) is False

    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        lambda event: calls.append(("replacement", event)) or True,
    )
    sip.delete(old)
    assert bus.publish_resource_lifecycle(request) is False
    assert [name for name, _event in calls] == ["old"]
    assert bus.acknowledge_resource_lifecycle_recipient(request, token) is True
    assert bus.publish_resource_lifecycle(request) is True
    assert [name for name, _event in calls] == ["old"]


def test_destroyed_resource_lifecycle_recipient_stays_exactly_pending(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()

    class Recipient(QObject):
        def receive(self, _request):
            return False

    recipient = Recipient()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "analysis",
        recipient.receive,
        owner=recipient,
    )
    request = ResourceLifecycleRequested(15, "disconnect-domains")
    assert bus.publish_resource_lifecycle(request) is False

    sip.delete(recipient)

    assert bus.publish_resource_lifecycle(request) is False
    assert bus.pending_resource_lifecycle_request_count == 1
    assert bus.acknowledge_resource_lifecycle_recipient(request, token) is True
    assert bus.publish_resource_lifecycle(request) is True


def test_resource_lifecycle_dispatcher_disconnect_abandons_bounded_pending():
    bus = SequenceEventBus()
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "pending", lambda _request: False
    )
    request = ResourceLifecycleRequested(17, "disconnect-domains")

    assert bus.publish_resource_lifecycle(request) is False
    assert bus.pending_resource_lifecycle_request_count == 1
    assert bus.resource_lifecycle_pending_limit > 0
    assert bus.resource_lifecycle_history_limit > 0

    bus.close_resource_lifecycle_dispatcher()

    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.abandoned_resource_lifecycle_request_count == 1
    assert bus.publish_resource_lifecycle(request) is False


def test_resource_lifecycle_reentrant_dispatcher_close_cannot_complete_delivery():
    bus = SequenceEventBus()
    request = ResourceLifecycleRequested(19, "disconnect-domains")

    def close_during_delivery(_request):
        bus.close_resource_lifecycle_dispatcher()
        return True

    bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "closing", close_during_delivery
    )

    assert bus.publish_resource_lifecycle(request) is False
    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.completed_resource_lifecycle_request_count == 0
    assert bus.abandoned_resource_lifecycle_request_count == 1


def test_resource_lifecycle_reentrant_recipient_replacement_cannot_ack_old_token(
    qapp,
):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    calls = []

    class Recipient(QObject):
        def receive(self, request):
            calls.append(("old", request.shutdown_generation))
            bus.register_resource_lifecycle_recipient(
                "disconnect-domains",
                "workflow",
                replacement.receive,
                owner=replacement,
            )
            return True

    class Replacement(QObject):
        def receive(self, request):
            calls.append(("replacement", request.shutdown_generation))
            return True

    old = Recipient()
    replacement = Replacement()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "workflow", old.receive, owner=old
    )
    request = ResourceLifecycleRequested(20, "disconnect-domains")

    assert bus.publish_resource_lifecycle(request) is False
    assert bus.completed_resource_lifecycle_request_count == 0
    assert bus.pending_resource_lifecycle_request_count == 1
    assert calls == [("old", 20)]

    sip.delete(old)
    assert bus.acknowledge_resource_lifecycle_recipient(request, token) is True
    assert bus.publish_resource_lifecycle(
        ResourceLifecycleRequested(21, "disconnect-domains")
    ) is True
    assert calls == [("old", 20), ("replacement", 21)]


def test_resource_lifecycle_retired_owner_resolution_is_exact(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()

    class Recipient(QObject):
        def receive(self, _request):
            return False

    recipient = Recipient()
    other = Recipient()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "analysis",
        recipient.receive,
        owner=recipient,
    )
    request = ResourceLifecycleRequested(21, "disconnect-domains")
    assert bus.publish_resource_lifecycle(request) is False
    assert bus.acknowledge_resource_lifecycle_recipient(request, token) is False

    sip.delete(recipient)
    assert bus.publish_resource_lifecycle(request) is False

    assert bus.resolve_retired_resource_lifecycle_recipient(
        request,
        token,
        owner_identity=id(other),
        registration_generation=token.version,
    ) is False
    assert bus.resolve_retired_resource_lifecycle_recipient(
        ResourceLifecycleRequested(22, "disconnect-domains"),
        token,
        owner_identity=id(recipient),
        registration_generation=token.version,
    ) is False
    assert bus.resolve_retired_resource_lifecycle_recipient(
        request,
        token,
        owner_identity=id(recipient),
        registration_generation=token.version + 1,
    ) is False
    assert bus.resolve_retired_resource_lifecycle_recipient(
        request,
        token,
        owner_identity=id(recipient),
        registration_generation=token.version,
    ) is True
    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.completed_resource_lifecycle_request_count == 1


def test_resource_lifecycle_rejects_equal_distinct_request_copies(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    calls = []
    retry_ready = {"value": False}

    class Recipient(QObject):
        def receive(self, request):
            calls.append(("retry", request))
            return retry_ready["value"]

    recipient = Recipient()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "retry",
        recipient.receive,
        owner=recipient,
    )
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "stable",
        lambda request: calls.append(("stable", request)) or True,
    )
    request = ResourceLifecycleRequested(25, "disconnect-domains")
    copies = (
        copy.copy(request),
        copy.deepcopy(request),
        pickle.loads(pickle.dumps(request)),
    )
    assert all(item == request and item is not request for item in copies)

    assert bus.publish_resource_lifecycle(request) is False
    assert [name for name, _request in calls] == ["retry", "stable"]
    for distinct in copies:
        assert bus.publish_resource_lifecycle(distinct) is False
    assert [name for name, _request in calls] == ["retry", "stable"]

    sip.delete(recipient)
    assert bus.publish_resource_lifecycle(request) is False
    for distinct in copies:
        assert bus.resolve_retired_resource_lifecycle_recipient(
            distinct,
            token,
            owner_identity=id(recipient),
            registration_generation=token.version,
        ) is False
        assert bus.pending_resource_lifecycle_request_count == 1

    assert bus.resolve_retired_resource_lifecycle_recipient(
        request,
        token,
        owner_identity=id(recipient),
        registration_generation=token.version,
    ) is True
    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.publish_resource_lifecycle(copies[0]) is False
    assert bus.publish_resource_lifecycle(request) is True


def test_resource_lifecycle_exact_identity_has_no_value_equality_fallback():
    import inspect

    source = inspect.getsource(SequenceEventBus)
    assert "delivery.request != request" not in source
    assert source.count("delivery.request is not request") >= 2


def test_retained_cleanup_registry_has_no_module_owned_mutable_creation_state():
    trees = tuple(
        ast.parse(inspect.getsource(module))
        for module in (event_bus_module, workflow_controller_module)
    )
    module_assignments = tuple(
        node
        for tree in trees
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    )
    assigned_names = {
        target.id
        for node in module_assignments
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }

    assert "_RETAINED_CLEANUP_REGISTRY_CREATION_LOCK" not in assigned_names
    assert all(
        not any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "RLock"
            for candidate in ast.walk(node)
        )
        for node in module_assignments
    )
    assert not any(
        isinstance(node, ast.ClassDef)
        and node.name == "_RetainedCleanupLifecycleRegistry"
        for node in trees[1].body
    )


def test_retained_cleanup_roots_are_isolated_and_first_registration_is_thread_safe():
    buses = (SequenceEventBus(), SequenceEventBus())
    assert (
        buses[0]._retained_cleanup_lifecycle_registry
        is not buses[1]._retained_cleanup_lifecycle_registry
    )
    barrier = Barrier(16)
    registrations = []

    def register(index):
        bus = buses[index % 2]
        token = object()
        lifecycle = object()
        barrier.wait()
        registrations.append(
            (
                bus,
                token,
                lifecycle,
                bus._register_retained_cleanup_lifecycle(token, lifecycle),
            )
        )

    workers = [Thread(target=register, args=(index,)) for index in range(16)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert len(registrations) == 16
    assert all(
        registered
        is RetainedCleanupLifecycleRegistrationResult.REGISTERED
        for *_identity, registered in registrations
    )
    assert buses[0]._retained_cleanup_lifecycle_count() == 8
    assert buses[1]._retained_cleanup_lifecycle_count() == 8
    for bus, token, lifecycle, _registered in registrations:
        assert bus._resolve_retained_cleanup_lifecycle(token) is lifecycle
        other_bus = buses[1] if bus is buses[0] else buses[0]
        assert other_bus._resolve_retained_cleanup_lifecycle(token) is None
        assert bus._retire_retained_cleanup_lifecycle(token, lifecycle) is True
    assert buses[0]._retained_cleanup_lifecycle_count() == 0
    assert buses[1]._retained_cleanup_lifecycle_count() == 0

    shared_token = object()
    first_lifecycle = object()
    second_lifecycle = object()
    assert buses[0]._register_retained_cleanup_lifecycle(
        shared_token, first_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert buses[1]._register_retained_cleanup_lifecycle(
        shared_token, second_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert buses[0]._register_retained_cleanup_lifecycle(
        shared_token, second_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
    assert buses[0]._resolve_retained_cleanup_lifecycle(
        shared_token
    ) is first_lifecycle
    assert buses[1]._resolve_retained_cleanup_lifecycle(
        shared_token
    ) is second_lifecycle


def test_retained_cleanup_registration_uses_exact_identity_without_token_code():
    class HostileToken:
        def __hash__(self):
            raise SystemExit("hash must not run")

        def __eq__(self, _other):
            raise SystemExit("equality must not run")

    bus = SequenceEventBus()
    first_token = HostileToken()
    second_token = HostileToken()
    first_lifecycle = object()
    replacement = object()

    assert bus._register_retained_cleanup_lifecycle(
        first_token, first_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert bus._register_retained_cleanup_lifecycle(
        first_token, first_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.IDEMPOTENT
    assert bus._register_retained_cleanup_lifecycle(
        first_token, replacement
    ) is RetainedCleanupLifecycleRegistrationResult.TOKEN_COLLISION
    assert bus._register_retained_cleanup_lifecycle(
        second_token, replacement
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert bus._resolve_retained_cleanup_lifecycle(
        first_token
    ) is first_lifecycle
    assert bus._resolve_retained_cleanup_lifecycle(second_token) is replacement
    assert bus._retire_retained_cleanup_lifecycle(
        first_token, replacement
    ) is False
    assert bus._retire_retained_cleanup_lifecycle(
        first_token, first_lifecycle
    ) is True


def test_retained_cleanup_roots_release_on_exact_bus_native_teardown_and_gc(qapp):
    from PyQt5 import sip

    class Lifecycle:
        pass

    retired_bus = SequenceEventBus()
    surviving_bus = SequenceEventBus()
    retired_token = object()
    surviving_token = object()
    retired_lifecycle = Lifecycle()
    surviving_lifecycle = Lifecycle()
    retired_bus_ref = ref(retired_bus)
    retired_lifecycle_ref = ref(retired_lifecycle)
    surviving_lifecycle_ref = ref(surviving_lifecycle)

    assert retired_bus._register_retained_cleanup_lifecycle(
        retired_token, retired_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED
    assert surviving_bus._register_retained_cleanup_lifecycle(
        surviving_token, surviving_lifecycle
    ) is RetainedCleanupLifecycleRegistrationResult.REGISTERED

    sip.delete(retired_bus)
    assert retired_bus._register_retained_cleanup_lifecycle(
        object(), object()
    ) is RetainedCleanupLifecycleRegistrationResult.CLOSED_NATIVE_DELETED
    del retired_lifecycle
    gc.collect()

    assert retired_lifecycle_ref() is None
    assert surviving_lifecycle_ref() is surviving_lifecycle
    assert surviving_bus._resolve_retained_cleanup_lifecycle(
        retired_token
    ) is None
    assert surviving_bus._resolve_retained_cleanup_lifecycle(
        surviving_token
    ) is surviving_lifecycle

    del retired_bus
    gc.collect()
    assert retired_bus_ref() is None
    assert surviving_bus._retire_retained_cleanup_lifecycle(
        surviving_token, surviving_lifecycle
    )


def test_resource_lifecycle_replacement_resolves_only_old_exact_token(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    calls = []

    class Recipient(QObject):
        def __init__(self, name, result):
            super().__init__()
            self.name = name
            self.result = result

        def receive(self, request):
            calls.append((self.name, request.shutdown_generation))
            return self.result

    old = Recipient("old", False)
    replacement = Recipient("replacement", True)
    old_token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "workflow", old.receive, owner=old
    )
    old_request = ResourceLifecycleRequested(23, "disconnect-domains")
    assert bus.publish_resource_lifecycle(old_request) is False

    sip.delete(old)
    new_token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        replacement.receive,
        owner=replacement,
    )
    assert bus.publish_resource_lifecycle(old_request) is False
    assert calls == [("old", 23)]
    assert bus.resolve_retired_resource_lifecycle_recipient(
        old_request,
        new_token,
        owner_identity=id(replacement),
        registration_generation=new_token.version,
    ) is False
    assert bus.resolve_retired_resource_lifecycle_recipient(
        old_request,
        old_token,
        owner_identity=id(old),
        registration_generation=old_token.version,
    ) is True

    new_request = ResourceLifecycleRequested(24, "disconnect-domains")
    assert bus.publish_resource_lifecycle(new_request) is True
    assert calls == [("old", 23), ("replacement", 24)]


def _registered_continuation_owner(bus):
    from PyQt5.QtCore import QObject

    owner = QObject()
    bus.register_workflow_continuation_lifecycle_owner(owner)
    return owner


def _continuation_message(*, label="OK", generation=0):
    return AnalysisTransportReady(
        "analysis", "source", "record", generation, {"Label": label}
    )


def _continuation_outcome(bus, owner, message=None, delivery_id=None):
    if message is None:
        message = _continuation_message()
    if delivery_id is None:
        delivery_id = ("analysis-transport", "analysis", 0)
    return bus.deliver_workflow_continuation_outcome(
        delivery_id,
        "analysis-transport",
        message,
        owner=owner,
    )


def _assert_bounded_non_payload_reason(outcome):
    assert type(outcome) is WorkflowContinuationDeliveryOutcome
    assert type(outcome.reason) is str
    assert outcome.reason
    assert len(outcome.reason) <= 256
    assert "OK" not in outcome.reason


def test_workflow_continuation_delivery_outcome_exact_types_and_bool_wrapper():
    assert tuple(WorkflowContinuationDeliveryStatus) == (
        WorkflowContinuationDeliveryStatus.ACK,
        WorkflowContinuationDeliveryStatus.RETRYABLE_NACK,
        WorkflowContinuationDeliveryStatus.PERMANENT_REJECT,
    )
    assert tuple(WorkflowContinuationRecipientResult) == (
        WorkflowContinuationRecipientResult.ACK,
        WorkflowContinuationRecipientResult.RETRYABLE_NACK,
        WorkflowContinuationRecipientResult.PERMANENT_REJECT,
    )

    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "capture", lambda _message: True
    )
    message = _continuation_message()
    delivery_id = ("analysis-transport", "analysis", 0)

    outcome = _continuation_outcome(bus, owner, message, delivery_id)

    assert outcome.status is WorkflowContinuationDeliveryStatus.ACK
    assert outcome.reason == ""
    wrapped = bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    )
    assert type(wrapped) is bool
    assert wrapped is True


@pytest.mark.parametrize(
    ("recipient_result", "expected_status", "expected_bool"),
    (
        (
            WorkflowContinuationRecipientResult.ACK,
            WorkflowContinuationDeliveryStatus.ACK,
            True,
        ),
        (
            WorkflowContinuationRecipientResult.RETRYABLE_NACK,
            WorkflowContinuationDeliveryStatus.RETRYABLE_NACK,
            False,
        ),
        (
            WorkflowContinuationRecipientResult.PERMANENT_REJECT,
            WorkflowContinuationDeliveryStatus.PERMANENT_REJECT,
            False,
        ),
    ),
)
def test_workflow_continuation_bool_wrapper_projects_detailed_status_exactly(
    recipient_result, expected_status, expected_bool
):
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", lambda _message: recipient_result
    )
    message = _continuation_message()

    outcome = _continuation_outcome(bus, owner, message)

    assert outcome.status is expected_status
    wrapped = bus.deliver_workflow_continuation(
        ("analysis-transport", "other-analysis", 0),
        "analysis-transport",
        message,
        owner=owner,
    )
    assert type(wrapped) is bool
    assert wrapped is expected_bool


def test_workflow_continuation_delivery_outcome_classifies_admission_branches():
    from PyQt5.QtCore import QObject

    message = _continuation_message()

    inactive_bus = SequenceEventBus()
    inactive_owner = _registered_continuation_owner(inactive_bus)
    inactive_bus.close_workflow_continuation_dispatcher()
    inactive = _continuation_outcome(inactive_bus, inactive_owner, message)

    unauthorized_bus = SequenceEventBus()
    unauthorized = _continuation_outcome(unauthorized_bus, QObject(), message)

    invalid_bus = SequenceEventBus()
    invalid_owner = _registered_continuation_owner(invalid_bus)
    unsupported_kind = invalid_bus.deliver_workflow_continuation_outcome(
        ("unknown",), "unknown", message, owner=invalid_owner
    )
    invalid_delivery_id = invalid_bus.deliver_workflow_continuation_outcome(
        ("wrong-kind",), "analysis-transport", message, owner=invalid_owner
    )
    wrong_message = invalid_bus.deliver_workflow_continuation_outcome(
        ("analysis-transport", "wrong-message"),
        "analysis-transport",
        object(),
        owner=invalid_owner,
    )
    unsupported_identity = _continuation_outcome(
        invalid_bus,
        invalid_owner,
        _continuation_message(label="x" * 20_000),
        ("analysis-transport", "oversized", 0),
    )

    for outcome in (
        inactive,
        unauthorized,
        unsupported_kind,
        invalid_delivery_id,
        wrong_message,
        unsupported_identity,
    ):
        assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
        _assert_bounded_non_payload_reason(outcome)


def test_workflow_continuation_delivery_outcome_classifies_history_and_capacity():
    completed_bus = SequenceEventBus()
    completed_owner = _registered_continuation_owner(completed_bus)
    completed_bus.register_workflow_continuation_recipient(
        "analysis-transport", "capture", lambda _message: True
    )
    message = _continuation_message()
    delivery_id = ("analysis-transport", "analysis", 0)
    assert _continuation_outcome(
        completed_bus, completed_owner, message, delivery_id
    ).status is WorkflowContinuationDeliveryStatus.ACK

    equal_completed = _continuation_outcome(
        completed_bus, completed_owner, _continuation_message(), delivery_id
    )
    conflicting_completed = _continuation_outcome(
        completed_bus,
        completed_owner,
        _continuation_message(label="different"),
        delivery_id,
    )
    cross_owner = _continuation_outcome(
        completed_bus,
        _registered_continuation_owner(completed_bus),
        message,
        delivery_id,
    )
    assert equal_completed.status is WorkflowContinuationDeliveryStatus.ACK
    assert equal_completed.reason == ""
    for outcome in (conflicting_completed, cross_owner):
        assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
        _assert_bounded_non_payload_reason(outcome)

    absent_bus = SequenceEventBus()
    absent_owner = _registered_continuation_owner(absent_bus)
    absent = _continuation_outcome(absent_bus, absent_owner)
    assert absent.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    _assert_bounded_non_payload_reason(absent)

    capacity_bus = SequenceEventBus()
    capacity_owner = _registered_continuation_owner(capacity_bus)
    capacity_bus.register_workflow_continuation_recipient(
        "analysis-transport", "blocked", lambda _message: False
    )
    for generation in range(capacity_bus._workflow_continuation_history_limit):
        outcome = _continuation_outcome(
            capacity_bus,
            capacity_owner,
            _continuation_message(generation=generation),
            ("analysis-transport", "analysis", generation),
        )
        assert outcome.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    capacity = _continuation_outcome(
        capacity_bus,
        capacity_owner,
        _continuation_message(generation=999),
        ("analysis-transport", "analysis", 999),
    )
    assert capacity.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    _assert_bounded_non_payload_reason(capacity)


@pytest.mark.parametrize(
    ("behavior", "expected_status"),
    (
        ("exception", WorkflowContinuationDeliveryStatus.RETRYABLE_NACK),
        (False, WorkflowContinuationDeliveryStatus.RETRYABLE_NACK),
        (None, WorkflowContinuationDeliveryStatus.ACK),
        (True, WorkflowContinuationDeliveryStatus.ACK),
    ),
)
def test_workflow_continuation_delivery_outcome_normalizes_recipient_results(
    behavior, expected_status
):
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)

    def recipient(_message):
        if behavior == "exception":
            raise RuntimeError("temporary")
        return behavior

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )
    outcome = _continuation_outcome(bus, owner)

    assert outcome.status is expected_status
    if expected_status is WorkflowContinuationDeliveryStatus.ACK:
        assert outcome.reason == ""
    else:
        _assert_bounded_non_payload_reason(outcome)


def test_workflow_continuation_delivery_outcome_recipient_absent_is_retryable():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "retired", lambda _message: False
    )
    message = _continuation_message()
    assert _continuation_outcome(bus, owner, message).status is (
        WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    )
    bus.unregister_workflow_continuation_recipient(
        "analysis-transport", "retired"
    )

    outcome = _continuation_outcome(bus, owner, message)

    assert token is not None
    assert outcome.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    _assert_bounded_non_payload_reason(outcome)


def test_workflow_continuation_recipient_permanent_reject_abandons_delivery():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "first",
        lambda _message: calls.append("first")
        or WorkflowContinuationRecipientResult.ACK,
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "terminal",
        lambda _message: calls.append("terminal")
        or WorkflowContinuationRecipientResult.PERMANENT_REJECT,
    )

    outcome = _continuation_outcome(bus, owner)
    retry = _continuation_outcome(
        bus,
        owner,
        _continuation_message(),
        ("analysis-transport", "analysis", 0),
    )

    assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    assert retry.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    _assert_bounded_non_payload_reason(retry)
    _assert_bounded_non_payload_reason(outcome)
    assert calls == ["first", "terminal"]
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1
    diagnostic = bus.abandoned_workflow_continuation_diagnostics[-1]
    assert diagnostic.reason == outcome.reason
    assert len(diagnostic.reason) <= 256


def test_workflow_continuation_pending_identity_conflict_is_abandoned_atomically():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "blocked",
        lambda message: calls.append(message.payload["Label"]) or False,
    )
    delivery_id = ("analysis-transport", "analysis", 0)
    original = _continuation_message()
    conflicting = _continuation_message(label="conflicting-payload")
    initial = _continuation_outcome(bus, owner, original, delivery_id)
    before = bus.abandoned_workflow_continuation_delivery_count

    outcome = _continuation_outcome(bus, owner, conflicting, delivery_id)
    retry_original = _continuation_outcome(
        bus, owner, _continuation_message(), delivery_id
    )
    retry_conflicting = _continuation_outcome(
        bus,
        owner,
        _continuation_message(label="conflicting-payload"),
        delivery_id,
    )

    assert initial.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    assert retry_original.status is (
        WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    )
    assert retry_conflicting.status is (
        WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    )
    _assert_bounded_non_payload_reason(retry_original)
    _assert_bounded_non_payload_reason(retry_conflicting)
    _assert_bounded_non_payload_reason(outcome)
    assert calls == ["OK"]
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == before + 1
    assert bus.abandoned_workflow_continuation_diagnostics[-1].reason == (
        outcome.reason
    )


def test_workflow_continuation_delivery_outcome_abandons_pending_cross_owner_collision():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)
    calls = []
    accept_b = False

    def recipient(message):
        calls.append(message.payload["Label"])
        return message.payload["Label"] == "A" or accept_b

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )
    delivery_id = ("analysis-transport", "shared", 0)
    message_a = _continuation_message(label="A")
    message_b = _continuation_message(label="B")

    first_b = _continuation_outcome(bus, owner_b, message_b, delivery_id)
    completed_a = _continuation_outcome(bus, owner_a, message_a, delivery_id)
    accept_b = True
    retried_b = _continuation_outcome(bus, owner_b, message_b, delivery_id)

    assert first_b.status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    assert completed_a.status is WorkflowContinuationDeliveryStatus.ACK
    assert retried_b.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    _assert_bounded_non_payload_reason(retried_b)
    assert calls == ["B", "A"]
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 1
    assert bus.abandoned_workflow_continuation_delivery_count == 1
    assert bus.abandoned_workflow_continuation_diagnostics[-1].reason == (
        retried_b.reason
    )


def test_workflow_continuation_async_ack_abandons_pending_cross_owner_collision():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)

    def recipient(message):
        return message.payload["Label"] == "A"

    recipient_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )
    delivery_id = ("analysis-transport", "shared", 0)
    message_a = _continuation_message(label="A")
    message_b = _continuation_message(label="B")
    assert _continuation_outcome(
        bus, owner_b, message_b, delivery_id
    ).status is WorkflowContinuationDeliveryStatus.RETRYABLE_NACK
    assert _continuation_outcome(
        bus, owner_a, message_a, delivery_id
    ).status is WorkflowContinuationDeliveryStatus.ACK

    acknowledged = bus.acknowledge_workflow_continuation_recipient(
        delivery_id,
        "analysis-transport",
        message_b,
        recipient_token,
    )

    assert acknowledged is False
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 1
    assert bus.abandoned_workflow_continuation_delivery_count == 1
    diagnostic = bus.abandoned_workflow_continuation_diagnostics[-1]
    assert diagnostic.reason
    assert len(diagnostic.reason) <= 256
    assert "A" not in diagnostic.reason
    assert "B" not in diagnostic.reason


def test_workflow_continuation_reentry_close_never_completes_stale_delivery():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)

    def recipient(_message):
        bus.close_workflow_continuation_dispatcher()
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )

    outcome = _continuation_outcome(bus, owner)

    assert outcome.status is not WorkflowContinuationDeliveryStatus.ACK
    _assert_bounded_non_payload_reason(outcome)
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_workflow_continuation_reentry_conflict_never_completes_stale_delivery():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    delivery_id = ("analysis-transport", "analysis", 0)
    nested_outcomes = []

    def recipient(message):
        assert message.payload["Label"] == "outer"
        nested_outcomes.append(
            _continuation_outcome(
                bus,
                owner,
                _continuation_message(label="nested-conflict"),
                delivery_id,
            )
        )
        nested_outcomes.append(
            _continuation_outcome(bus, owner, message, delivery_id)
        )
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )

    outcome = _continuation_outcome(
        bus,
        owner,
        _continuation_message(label="outer"),
        delivery_id,
    )

    assert nested_outcomes[0].status is (
        WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    )
    assert nested_outcomes[1] == WorkflowContinuationDeliveryOutcome(
        WorkflowContinuationDeliveryStatus.PERMANENT_REJECT,
        "continuation delivery was abandoned",
    )
    assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    _assert_bounded_non_payload_reason(outcome)
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_workflow_continuation_reentry_same_delivery_defers_to_terminal_result():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    message = _continuation_message()
    delivery_id = ("analysis-transport", "analysis", 0)
    nested_outcomes = []

    def recipient(_message):
        nested_outcomes.append(
            _continuation_outcome(bus, owner, message, delivery_id)
        )
        return WorkflowContinuationRecipientResult.PERMANENT_REJECT

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )

    outcome = _continuation_outcome(bus, owner, message, delivery_id)

    assert nested_outcomes == [
        WorkflowContinuationDeliveryOutcome(
            WorkflowContinuationDeliveryStatus.RETRYABLE_NACK,
            "continuation delivery is already active",
        )
    ]
    assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    _assert_bounded_non_payload_reason(outcome)
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_workflow_continuation_reentry_cross_owner_completion_abandons_outer():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)
    delivery_id = ("analysis-transport", "shared", 0)
    message_a = _continuation_message(label="A")
    message_b = _continuation_message(label="B")
    nested_outcomes = []

    def recipient(message):
        if message.payload["Label"] == "B":
            nested_outcomes.append(
                _continuation_outcome(
                    bus, owner_a, message_a, delivery_id
                )
            )
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "recipient", recipient
    )

    outcome = _continuation_outcome(bus, owner_b, message_b, delivery_id)

    assert nested_outcomes[0].status is WorkflowContinuationDeliveryStatus.ACK
    assert outcome.status is WorkflowContinuationDeliveryStatus.PERMANENT_REJECT
    _assert_bounded_non_payload_reason(outcome)
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 1
    assert bus.abandoned_workflow_continuation_delivery_count == 1


def test_workflow_continuation_reentry_different_kind_remains_synchronous():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    nested_outcomes = []
    label_message = messages.CommitRecordingLabelRequested(
        "command", "record", "OK", ()
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "label", lambda _message: True
    )

    def transport(_message):
        nested_outcomes.append(
            bus.deliver_workflow_continuation_outcome(
                ("label-commit", "command", 0),
                "label-commit",
                label_message,
                owner=owner,
            )
        )
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "transport", transport
    )

    outcome = _continuation_outcome(bus, owner)

    assert nested_outcomes == [
        WorkflowContinuationDeliveryOutcome(
            WorkflowContinuationDeliveryStatus.ACK
        )
    ]
    assert outcome.status is WorkflowContinuationDeliveryStatus.ACK
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 2


def test_cancel_imported_audio_message_and_command_channel_are_exact():
    command = messages.CancelImportedAudioRequested(
        "import-1", 7, "operator cancelled"
    )
    bus = SequenceEventBus()
    captured = []
    bus.commands.cancel_imported_audio_requested.connect(captured.append)

    bus.commands.cancel_imported_audio_requested.emit(command)

    assert captured == [command]
    assert command.import_id == "import-1"
    assert command.workflow_generation == 7
    with pytest.raises(ValueError):
        messages.CancelImportedAudioRequested("", 7, "cancel")
    with pytest.raises(ValueError):
        messages.CancelImportedAudioRequested("import-1", -1, "cancel")


def test_import_terminal_dispatch_retries_critical_before_noncritical_observer():
    bus = SequenceEventBus()
    calls = []
    accepted = {"critical": False}

    def critical(message):
        calls.append(("critical", message))
        if not accepted["critical"]:
            raise KeyboardInterrupt("workflow unavailable")
        return True

    def hostile_observer(message):
        calls.append(("observer", message))
        raise SystemExit("observer failed")

    bus.register_import_terminal_recipient(
        "observer", hostile_observer, critical=False
    )
    bus.register_import_terminal_recipient(
        "workflow", critical, critical=True
    )
    event = messages.ImportedAudioFailed("import-1", "cancelled")
    delivery_id = ("ImportedAudioFailed", "import-1")

    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert calls == [("critical", event)]
    accepted["critical"] = True
    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert calls == [
        ("critical", event),
        ("critical", event),
        ("observer", event),
    ]
    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert len(calls) == 3


def test_import_terminal_exact_synchronous_reentry_never_reinvokes_recipient():
    bus = SequenceEventBus()
    calls = []
    nested_results = []
    event = messages.ImportedAudioReady(
        "import-reentrant", {"record_id": "record-reentrant"}
    )
    delivery_id = ("ImportedAudioReady", "import-reentrant")

    def workflow(message):
        calls.append(message)
        if len(calls) == 1:
            nested_results.append(
                bus.deliver_import_terminal(delivery_id, message)
            )
        return True

    bus.register_import_terminal_recipient(
        "workflow", workflow, critical=True
    )

    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert nested_results == [False]
    assert calls == [event]
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("ordinary"),
        KeyboardInterrupt("interrupt"),
        SystemExit("exit"),
    ],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_import_terminal_reentry_gate_resets_after_recipient_baseexception(failure):
    bus = SequenceEventBus()
    calls = []
    nested_results = []
    event = messages.ImportedAudioFailed("import-exception", "failed")
    delivery_id = ("ImportedAudioFailed", "import-exception")

    def workflow(message):
        calls.append(message)
        if len(calls) == 1:
            nested_results.append(
                bus.deliver_import_terminal(delivery_id, message)
            )
            raise failure
        return ImportTerminalRecipientResult.ACK

    bus.register_import_terminal_recipient(
        "workflow", workflow, critical=True
    )

    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert nested_results == [False]
    assert calls == [event]
    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert calls == [event, event]


def test_import_terminal_collisions_do_not_exhaust_pending_capacity():
    bus = SequenceEventBus()
    observed = []
    bus.register_import_terminal_recipient(
        "workflow",
        lambda event: observed.append(event) or False,
        critical=True,
    )
    claimed = messages.ImportedAudioReady(
        "import-collision", {"record_id": "record-first"}
    )
    claimed_id = ("ImportedAudioReady", "import-collision")

    assert bus.deliver_import_terminal(claimed_id, claimed) is False
    for index in range(128):
        conflicting = (
            messages.ImportedAudioFailed(
                "import-collision", f"conflicting failure {index}"
            )
            if index % 2
            else messages.ImportedAudioReady(
                "import-collision", {"record_id": f"conflict-{index}"}
            )
        )
        assert bus.deliver_import_terminal(
            (type(conflicting).__name__, conflicting.import_id), conflicting
        ) is False

    assert bus.pending_import_terminal_delivery_count == 1
    assert observed == [claimed]

    next_event = messages.ImportedAudioFailed("import-next", "legitimate")
    assert bus.deliver_import_terminal(
        ("ImportedAudioFailed", "import-next"), next_event
    ) is False
    assert bus.pending_import_terminal_delivery_count == 2
    assert observed == [claimed, next_event]


def test_import_terminal_capacity_wait_preserves_first_exact_outcome():
    bus = SequenceEventBus()
    observed = []
    accepted_import_ids = set()

    def workflow(event):
        observed.append(event)
        return event.import_id in accepted_import_ids

    bus.register_import_terminal_recipient(
        "workflow", workflow, critical=True
    )
    retained = []
    for index in range(bus.import_terminal_pending_limit):
        event = messages.ImportedAudioFailed(f"active-{index}", "pending")
        delivery_id = ("ImportedAudioFailed", event.import_id)
        retained.append((delivery_id, event))
        assert bus.deliver_import_terminal(delivery_id, event) is False

    first = messages.ImportedAudioReady(
        "capacity-wait", {"record_id": "first-outcome"}
    )
    first_id = ("ImportedAudioReady", first.import_id)
    conflict = messages.ImportedAudioFailed(
        "capacity-wait", "conflicting outcome"
    )
    conflict_id = ("ImportedAudioFailed", conflict.import_id)

    assert bus.deliver_import_terminal(first_id, first) is False
    assert first not in observed
    assert bus.waiting_import_terminal_delivery_count == 1
    assert bus.import_terminal_claim_count == (
        bus.import_terminal_pending_limit + 1
    )
    assert bus.import_terminal_claim_count <= bus.import_terminal_claim_limit
    assert bus.abandon_import_terminal(
        retained[0][0], "release-capacity"
    ) is True

    assert bus.deliver_import_terminal(conflict_id, conflict) is False
    assert conflict not in observed
    assert bus.pending_import_terminal_delivery_count == (
        bus.import_terminal_pending_limit - 1
    )

    accepted_import_ids.add(first.import_id)
    assert bus.deliver_import_terminal(first_id, first) is True
    assert observed[-1] is first
    assert bus.waiting_import_terminal_delivery_count == 0
    assert bus.pending_import_terminal_delivery_count == (
        bus.import_terminal_pending_limit - 1
    )
    assert bus.deliver_import_terminal(conflict_id, conflict) is False
    assert conflict not in observed

    next_event = messages.ImportedAudioFailed("after-capacity-wait", "pending")
    assert bus.deliver_import_terminal(
        ("ImportedAudioFailed", next_event.import_id), next_event
    ) is False
    assert observed[-1] is next_event


def test_import_terminal_capacity_wait_can_be_explicitly_abandoned():
    bus = SequenceEventBus()
    observed = []
    bus.register_import_terminal_recipient(
        "workflow",
        lambda event: observed.append(event) or False,
        critical=True,
    )
    retained = []
    for index in range(bus.import_terminal_pending_limit):
        event = messages.ImportedAudioFailed(f"active-{index}", "pending")
        delivery_id = ("ImportedAudioFailed", event.import_id)
        retained.append((delivery_id, event))
        assert bus.deliver_import_terminal(delivery_id, event) is False

    abandoned = messages.ImportedAudioReady(
        "abandoned-capacity-wait", {"record_id": "first-outcome"}
    )
    abandoned_id = ("ImportedAudioReady", abandoned.import_id)
    assert bus.deliver_import_terminal(abandoned_id, abandoned) is False
    assert bus.waiting_import_terminal_delivery_count == 1
    assert bus.abandon_import_terminal(
        abandoned_id, "producer-abandoned"
    ) is True
    assert bus.import_terminal_abandonment_reason(abandoned_id) == (
        "producer-abandoned"
    )
    assert bus.waiting_import_terminal_delivery_count == 0

    conflict = messages.ImportedAudioFailed(
        abandoned.import_id, "must not replace abandoned outcome"
    )
    assert bus.deliver_import_terminal(
        ("ImportedAudioFailed", conflict.import_id), conflict
    ) is False
    assert conflict not in observed

    next_event = messages.ImportedAudioReady(
        "next-capacity-wait", {"record_id": "next"}
    )
    next_id = ("ImportedAudioReady", next_event.import_id)
    assert bus.deliver_import_terminal(next_id, next_event) is False
    assert next_event not in observed
    assert bus.waiting_import_terminal_delivery_count == 1
    assert bus.abandon_import_terminal(
        retained[0][0], "release-capacity"
    ) is True
    assert bus.deliver_import_terminal(next_id, next_event) is False
    assert observed[-1] is next_event


def test_import_terminal_permanent_reject_is_abandoned_without_retry_capacity():
    bus = SequenceEventBus()
    calls = []
    bus.register_import_terminal_recipient(
        "workflow",
        lambda event: calls.append(event)
        or ImportTerminalRecipientResult.PERMANENT_REJECT,
        critical=True,
    )
    rejected = messages.ImportedAudioFailed("import-rejected", "stale")
    delivery_id = ("ImportedAudioFailed", "import-rejected")

    assert bus.deliver_import_terminal(delivery_id, rejected) is False
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.import_terminal_abandonment_reason(delivery_id) == (
        "recipient-permanent-reject"
    )
    assert bus.deliver_import_terminal(delivery_id, rejected) is False
    assert calls == [rejected]


def test_import_terminal_noncritical_only_never_acknowledges_completion():
    bus = SequenceEventBus()
    observed = []
    bus.register_import_terminal_recipient(
        "observer", lambda event: observed.append(event) or True, critical=False
    )
    event = messages.ImportedAudioReady("import-1", {"record_id": "record"})
    delivery_id = ("ImportedAudioReady", "import-1")

    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert observed == []
    assert delivery_id not in bus._import_terminal_completed


def test_import_terminal_waits_for_every_required_critical_ack():
    bus = SequenceEventBus()
    calls = []
    second_ready = {"value": False}

    def first(event):
        calls.append(("first", event))
        return True

    def second(event):
        calls.append(("second", event))
        return second_ready["value"]

    bus.register_import_terminal_recipient("first", first, critical=True)
    bus.register_import_terminal_recipient("second", second, critical=True)
    event = messages.ImportedAudioReady("import-1", {"record_id": "record"})
    delivery_id = ("ImportedAudioReady", "import-1")

    assert bus.deliver_import_terminal(delivery_id, event) is False
    second_ready["value"] = True
    assert bus.deliver_import_terminal(delivery_id, event) is True
    assert calls == [("first", event), ("second", event), ("second", event)]


def test_abandoned_import_terminal_cannot_be_retried_as_success():
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    event = messages.ImportedAudioFailed("import-1", "failed")
    delivery_id = ("ImportedAudioFailed", "import-1")
    assert bus.deliver_import_terminal(delivery_id, event) is False

    assert bus.abandon_import_terminal(delivery_id, "analysis-disconnect") is True
    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert bus.import_terminal_abandonment_reason(delivery_id) == "analysis-disconnect"


def test_import_terminal_delivery_binds_exact_event_before_and_after_completion():
    bus = SequenceEventBus()
    accepted = {"value": False}
    observed = []
    bus.register_import_terminal_recipient(
        "workflow",
        lambda event: observed.append(event) or accepted["value"],
        critical=True,
    )
    original = messages.ImportedAudioReady(
        "import-1", {"record_id": "record-1"}
    )
    same_fields = messages.ImportedAudioReady(
        "import-1", {"record_id": "record-1"}
    )
    different_payload = messages.ImportedAudioReady(
        "import-1", {"record_id": "record-2"}
    )
    wrong_type = messages.ImportedAudioFailed("import-1", "failed")
    delivery_id = ("ImportedAudioReady", "import-1")

    assert bus.deliver_import_terminal(delivery_id, original) is False
    assert bus.deliver_import_terminal(delivery_id, same_fields) is False
    assert bus.deliver_import_terminal(delivery_id, different_payload) is False
    assert bus.deliver_import_terminal(delivery_id, wrong_type) is False
    assert observed == [original]

    accepted["value"] = True
    assert bus.deliver_import_terminal(delivery_id, original) is True
    assert bus.deliver_import_terminal(delivery_id, original) is True
    assert bus.deliver_import_terminal(delivery_id, same_fields) is False
    assert bus.deliver_import_terminal(delivery_id, different_payload) is False
    assert bus.deliver_import_terminal(delivery_id, wrong_type) is False
    assert observed == [original, original]


def test_import_terminal_rejects_delivery_id_that_disagrees_with_event():
    bus = SequenceEventBus()
    observed = []
    bus.register_import_terminal_recipient(
        "workflow", lambda event: observed.append(event) or True, critical=True
    )
    event = messages.ImportedAudioReady("import-1", {"record_id": "record"})

    assert bus.deliver_import_terminal(("ImportedAudioFailed", "import-1"), event) is False
    assert bus.deliver_import_terminal(("ImportedAudioReady", "import-2"), event) is False
    assert observed == []
    assert bus.pending_import_terminal_delivery_count == 0


def test_import_terminal_replacement_and_unregister_do_not_inherit_pending_ack():
    bus = SequenceEventBus()
    first_calls = []
    replacement_calls = []
    event = messages.ImportedAudioFailed("import-1", "failed")
    delivery_id = ("ImportedAudioFailed", "import-1")

    def first(message):
        first_calls.append(message)
        return False

    def replacement(message):
        replacement_calls.append(message)
        return True

    bus.register_import_terminal_recipient("workflow", first, critical=True)
    assert bus.deliver_import_terminal(delivery_id, event) is False
    bus.register_import_terminal_recipient(
        "workflow", replacement, critical=True
    )
    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert first_calls == [event]
    assert replacement_calls == []

    bus.unregister_import_terminal_recipient("workflow", replacement)
    bus.register_import_terminal_recipient("workflow", first, critical=True)
    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert first_calls == [event]


def test_import_terminal_qobject_recipient_destruction_keeps_pending_unacked():
    from PyQt5.QtCore import QCoreApplication, QEvent, QObject

    app = QCoreApplication.instance() or QCoreApplication([])
    bus = SequenceEventBus()
    calls = []

    class Recipient(QObject):
        def deliver(self, event):
            calls.append(event)
            return False

    owner = Recipient()
    bus.register_import_terminal_recipient(
        "workflow", owner.deliver, critical=True
    )
    event = messages.ImportedAudioFailed("import-1", "failed")
    delivery_id = ("ImportedAudioFailed", "import-1")
    assert bus.deliver_import_terminal(delivery_id, event) is False

    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    del owner
    gc.collect()

    assert bus.import_terminal_recipient_count == 0
    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert calls == [event]
    assert bus.pending_import_terminal_delivery_count == 1


def test_import_terminal_bound_recipient_gc_does_not_ack_pending():
    bus = SequenceEventBus()

    class Recipient:
        def deliver(self, _event):
            return False

    recipient = Recipient()
    bus.register_import_terminal_recipient(
        "workflow", recipient.deliver, critical=True
    )
    event = messages.ImportedAudioFailed("import-1", "failed")
    delivery_id = ("ImportedAudioFailed", "import-1")
    assert bus.deliver_import_terminal(delivery_id, event) is False

    del recipient
    gc.collect()

    assert bus.import_terminal_recipient_count == 0
    assert bus.deliver_import_terminal(delivery_id, event) is False
    assert bus.pending_import_terminal_delivery_count == 1


def test_import_terminal_recipient_and_pending_capacity_never_evicts_active():
    bus = SequenceEventBus()
    callbacks = []
    for index in range(bus.import_terminal_recipient_limit):
        callback = lambda _event, _index=index: False
        callbacks.append(callback)
        bus.register_import_terminal_recipient(
            f"recipient-{index}", callback, critical=index == 0
        )
    with pytest.raises(RuntimeError, match="recipient registry is full"):
        bus.register_import_terminal_recipient(
            "overflow", lambda _event: True, critical=True
        )

    retained = []
    for index in range(bus.import_terminal_pending_limit):
        event = messages.ImportedAudioFailed(f"import-{index}", "failed")
        retained.append((event, ("ImportedAudioFailed", f"import-{index}")))
        assert bus.deliver_import_terminal(retained[-1][1], event) is False
    overflow = messages.ImportedAudioFailed("import-overflow", "failed")
    assert bus.deliver_import_terminal(
        ("ImportedAudioFailed", "import-overflow"), overflow
    ) is False
    assert bus.pending_import_terminal_delivery_count == bus.import_terminal_pending_limit
    assert retained[0][1] in bus._import_terminal_pending


def test_import_terminal_completion_and_abandonment_histories_are_bounded():
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: True, critical=True
    )
    for index in range(1_000):
        event = messages.ImportedAudioFailed(f"complete-{index}", "failed")
        assert bus.deliver_import_terminal(
            ("ImportedAudioFailed", f"complete-{index}"), event
        ) is True
    assert (
        bus.completed_import_terminal_delivery_count
        == bus.import_terminal_history_limit
    )

    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    for index in range(1_000):
        event = messages.ImportedAudioFailed(f"abandon-{index}", "failed")
        delivery_id = ("ImportedAudioFailed", f"abandon-{index}")
        assert bus.deliver_import_terminal(delivery_id, event) is False
        assert bus.abandon_import_terminal(delivery_id, "test-retirement") is True
    assert (
        bus.abandoned_import_terminal_delivery_count
        == bus.import_terminal_history_limit
    )
    assert len(bus._import_terminal_claims) <= (
        bus.pending_import_terminal_delivery_count
        + bus.completed_import_terminal_delivery_count
        + bus.abandoned_import_terminal_delivery_count
    )


def test_continuation_composite_owner_and_message_identity_is_fail_closed():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)
    from PyQt5.QtCore import QObject

    unregistered_owner = QObject()
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "capture",
        lambda message: calls.append(message) or True,
    )
    delivery_id = ("analysis-transport", "shared", 0)
    original = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    logical_copy = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    different = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "NG"}
    )

    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        original,
        owner=owner_a,
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        logical_copy,
        owner=owner_a,
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        different,
        owner=owner_a,
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        original,
        owner=owner_b,
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        different,
        owner=owner_b,
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        original,
        owner=unregistered_owner,
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "label-commit",
        original,
        owner=owner_a,
    ) is False
    assert calls == [original]


def test_pending_composite_keys_allow_two_owners_and_exact_abandonment():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)
    accepting = False
    calls = []

    def recipient(message):
        calls.append(message.payload["Label"])
        return accepting

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "capture", recipient
    )
    delivery_id = ("analysis-transport", "shared", 0)
    message_a = AnalysisTransportReady(
        "analysis-a", "source", "record", 0, {"Label": "A"}
    )
    message_b = AnalysisTransportReady(
        "analysis-b", "source", "record", 0, {"Label": "B"}
    )
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        message_a,
        owner=owner_a,
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        message_b,
        owner=owner_b,
    ) is False
    assert bus.pending_workflow_continuation_delivery_count == 2

    assert bus.abandon_workflow_continuations(
        (delivery_id,), owner=owner_a, reason="workflow-cancellation"
    ) == 1
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert bus.abandon_workflow_continuations(
        (delivery_id,), owner=owner_a, reason="workflow-cancellation"
    ) == 0
    assert bus.pending_workflow_continuation_delivery_count == 1

    accepting = True
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        message_b,
        owner=owner_b,
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 1
    assert calls == ["A", "B", "B"]


def test_terminal_recipient_ack_requires_one_exact_delivery_and_opaque_token():
    bus = SequenceEventBus()
    owner_a = _registered_continuation_owner(bus)
    owner_b = _registered_continuation_owner(bus)
    calls = {"target": 0, "other": 0}
    other_accepting = False

    def target(_message):
        calls["target"] += 1
        return False

    def other(_message):
        calls["other"] += 1
        return other_accepting

    target_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "target", target
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "other", other
    )
    delivery_id = ("analysis-transport", "analysis", "source", "record", 0)
    message = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    logical_copy = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    for owner in (owner_a, owner_b):
        assert bus.deliver_workflow_continuation(
            delivery_id,
            "analysis-transport",
            message,
            owner=owner,
        ) is False

    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, target_token
    ) is False
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", logical_copy, target_token
    ) is False
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, object()
    ) is False
    assert bus.abandon_workflow_continuations(
        (delivery_id,), owner=owner_b, reason="owner-retired"
    ) == 1

    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, target_token
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 1
    other_accepting = True
    assert bus.deliver_workflow_continuation(
        delivery_id,
        "analysis-transport",
        message,
        owner=owner_a,
    ) is True
    assert calls == {"target": 2, "other": 3}
    assert bus.pending_workflow_continuation_delivery_count == 0


def test_terminal_recipient_ack_rejects_structural_and_copied_token_forgeries():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = {"target": 0, "other": 0}

    def target(_message):
        calls["target"] += 1
        return False

    def other(_message):
        calls["other"] += 1
        return False

    target_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "target", target
    )
    other_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "other", other
    )
    foreign_bus = SequenceEventBus()
    foreign_token = foreign_bus.register_workflow_continuation_recipient(
        "analysis-transport", "target", lambda _message: False
    )
    token_type = type(target_token)
    structural_copy = token_type(
        target_token.kind, target_token.name, target_token.version
    )
    delivery_id = ("analysis-transport", "analysis", "source", "record", 0)
    message = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )

    assert structural_copy is not target_token
    assert structural_copy != target_token
    assert (
        structural_copy.kind,
        structural_copy.name,
        structural_copy.version,
    ) == (
        target_token.kind,
        target_token.name,
        target_token.version,
    )
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    ) is False

    forged_tokens = (
        structural_copy,
        copy.deepcopy(target_token),
        pickle.loads(pickle.dumps(target_token)),
        foreign_token,
        token_type(target_token.kind, "other", target_token.version),
        token_type(target_token.kind, target_token.name, other_token.version),
        token_type("label-commit", target_token.name, target_token.version),
    )
    for forged_token in forged_tokens:
        assert forged_token is not target_token
        assert bus.acknowledge_workflow_continuation_recipient(
            delivery_id,
            "analysis-transport",
            message,
            forged_token,
        ) is False

    assert bus.pending_workflow_continuation_delivery_count == 1
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, target_token
    ) is True
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, target_token
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 1

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    ) is False
    assert calls == {"target": 1, "other": 2}
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, other_token
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 0


def test_replaced_recipient_token_cannot_ack_a_new_delivery():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    old_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "transport", lambda _message: False
    )
    new_token = bus.register_workflow_continuation_recipient(
        "analysis-transport", "transport", lambda _message: False
    )
    delivery_id = ("analysis-transport", "analysis", "source", "record", 0)
    message = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    ) is False
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, old_token
    ) is False
    assert bus.acknowledge_workflow_continuation_recipient(
        delivery_id, "analysis-transport", message, new_token
    ) is True
    assert bus.pending_workflow_continuation_delivery_count == 0


def test_hostile_or_oversized_completed_identity_fails_closed_without_callbacks():
    class Hostile:
        def __repr__(self):
            raise SystemExit("repr must not run")

        def __hash__(self):
            raise SystemExit("hash must not run")

        def __eq__(self, _other):
            raise SystemExit("equality must not run")

    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = {"count": 0}
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "capture",
        lambda _message: calls.__setitem__("count", calls["count"] + 1) or True,
    )
    hostile = object.__new__(AnalysisTransportReady)
    for name, value in (
        ("analysis_id", "analysis"),
        ("source_id", "source"),
        ("record_id", "record"),
        ("workflow_generation", 0),
        ("payload", Hostile()),
    ):
        object.__setattr__(hostile, name, value)
    oversized = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "x" * 20_000}
    )

    hostile_result = bus.deliver_workflow_continuation(
        ("analysis-transport", "hostile", 0),
        "analysis-transport",
        hostile,
        owner=owner,
    )
    oversized_result = bus.deliver_workflow_continuation(
        ("analysis-transport", "oversized", 0),
        "analysis-transport",
        oversized,
        owner=owner,
    )
    del hostile
    assert hostile_result is False
    assert oversized_result is False
    assert calls == {"count": 0}


def test_continuation_identity_accepts_equal_immutable_arrays_and_rejects_changes():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "array-capture",
        lambda message: calls.append(message) or True,
    )
    original = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.array([1.0, 2.0], dtype=np.float64)},
    )
    logical_copy = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.array([1.0, 2.0], dtype=np.float64)},
    )
    changed = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.array([1.0, 3.0], dtype=np.float64)},
    )
    delivery_id = ("analysis-transport", "array", 0)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", original, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", logical_copy, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", changed, owner=owner
    ) is False
    assert calls == [original]


@pytest.mark.parametrize("changed_dimension", ("dtype", "shape"))
def test_continuation_identity_immutable_arrays_reject_dtype_or_shape_changes(
    changed_dimension,
):
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "array-capture",
        lambda message: calls.append(message) or True,
    )
    source = np.array([1, 2], dtype=np.uint32)
    if changed_dimension == "dtype":
        changed_array = np.frombuffer(source.tobytes(), dtype=np.float32).copy()
    else:
        changed_array = source.reshape((1, 2))
    original = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"curve": source}
    )
    changed = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"curve": changed_array}
    )
    delivery_id = ("analysis-transport", "array-dimension", changed_dimension)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", original, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", changed, owner=owner
    ) is False
    assert calls == [original]


def test_continuation_identity_immutable_arrays_reject_untrusted_array_forms():
    hostile_calls = []

    class HostileArray(np.ndarray):
        def __repr__(self):
            hostile_calls.append("repr")
            raise AssertionError("hostile repr must not run")

        def __hash__(self):
            hostile_calls.append("hash")
            raise AssertionError("hostile hash must not run")

        def __eq__(self, _other):
            hostile_calls.append("eq")
            raise AssertionError("hostile equality must not run")

        def __iter__(self):
            hostile_calls.append("iter")
            raise AssertionError("hostile iteration must not run")

    ordinary = np.array([1.0], dtype=np.float64)
    hostile = ordinary.view(HostileArray)
    object_dtype = np.array([object()], dtype=object).view(
        messages._ImmutablePayloadArray
    )
    writable = np.ndarray.__new__(
        messages._ImmutablePayloadArray,
        shape=(1,),
        dtype=np.float64,
        buffer=bytearray(8),
    )
    unproven = ordinary.view(messages._ImmutablePayloadArray)

    for value in (ordinary, hostile, object_dtype, writable, unproven):
        budget = event_bus_module._ContinuationIdentityBudget()
        with pytest.raises(event_bus_module._UnsupportedContinuationIdentity):
            event_bus_module._bounded_continuation_value_identity(
                value, 0, budget
            )
    assert hostile_calls == []


def test_continuation_identity_production_shape_node_budget_is_bounded():
    assert event_bus_module._CANONICAL_IDENTITY_MAX_NODES == 8_192
    assert (
        event_bus_module._CANONICAL_IDENTITY_MAX_ARRAY_BYTES
        == 268_435_456
    )
    legacy_configuration = {
        f"feature-{index}": index for index in range(400)
    }
    message = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {
            "result_snapshot": legacy_configuration,
            "export_handoff": legacy_configuration,
            "request_configuration": legacy_configuration,
        },
    )
    budget = event_bus_module._ContinuationIdentityBudget()
    event_bus_module._bounded_continuation_value_identity(
        message.payload, 0, budget
    )
    assert 2_048 < budget.nodes < 8_192
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "capture",
        lambda delivered: calls.append(delivered) or True,
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "production-shape", 0),
        "analysis-transport",
        message,
        owner=owner,
    ) is True
    assert calls == [message]


def test_continuation_identity_array_byte_budget_counts_each_occurrence(
    monkeypatch,
):
    frozen_source = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.array([1.0, 2.0], dtype=np.float64)},
    ).payload["curve"]
    byte_limit = int(np.ndarray.nbytes.__get__(frozen_source))
    monkeypatch.setattr(
        event_bus_module,
        "_CANONICAL_IDENTITY_MAX_ARRAY_BYTES",
        byte_limit,
        raising=False,
    )
    one_occurrence = AnalysisTransportReady(
        "analysis", "source", "record", 0, {"first": frozen_source}
    )
    two_occurrences = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"first": frozen_source, "second": frozen_source},
    )
    assert two_occurrences.payload["first"] is two_occurrences.payload["second"]
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "capture",
        lambda delivered: calls.append(delivered) or True,
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "one-array", 0),
        "analysis-transport",
        one_occurrence,
        owner=owner,
    ) is True
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "two-arrays", 0),
        "analysis-transport",
        two_occurrences,
        owner=owner,
    ) is False
    assert calls == [one_occurrence]


def test_continuation_identity_immutable_arrays_use_one_structural_node():
    message = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.arange(20_000, dtype=np.float64)},
    )
    frozen_array = message.payload["curve"]
    budget = event_bus_module._ContinuationIdentityBudget()
    identity = event_bus_module._bounded_continuation_value_identity(
        frozen_array, 0, budget
    )
    expected_digest = hashlib.blake2b(
        memoryview(frozen_array).cast("B"),
        digest_size=32,
        person=b"seq-cont-v1",
    ).digest()
    assert identity == (
        "immutable-ndarray",
        np.dtype.str.__get__(np.ndarray.dtype.__get__(frozen_array)).encode(
            "utf-8"
        ),
        (20_000,),
        expected_digest,
    )
    assert budget.nodes == 1
    assert budget.array_bytes == 20_000 * np.dtype(np.float64).itemsize
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "capture",
        lambda delivered: calls.append(delivered) or True,
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "large-array", 0),
        "analysis-transport",
        message,
        owner=owner,
    ) is True
    assert calls == [message]


def test_continuation_identity_immutable_arrays_support_empty_shapes():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "array-capture",
        lambda message: calls.append(message) or True,
    )
    original = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.empty((0, 2), dtype=np.float64)},
    )
    logical_copy = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.empty((0, 2), dtype=np.float64)},
    )
    changed_dtype = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.empty((0, 2), dtype=np.int64)},
    )
    changed_shape = AnalysisTransportReady(
        "analysis",
        "source",
        "record",
        0,
        {"curve": np.empty((0, 3), dtype=np.float64)},
    )
    frozen_array = original.payload["curve"]
    budget = event_bus_module._ContinuationIdentityBudget()
    assert event_bus_module._bounded_continuation_value_identity(
        frozen_array, 0, budget
    ) == (
        "immutable-ndarray",
        np.dtype.str.__get__(np.ndarray.dtype.__get__(frozen_array)).encode(
            "utf-8"
        ),
        (0, 2),
        hashlib.blake2b(
            b"", digest_size=32, person=b"seq-cont-v1"
        ).digest(),
    )
    assert budget.nodes == 1
    assert budget.array_bytes == 0
    delivery_id = ("analysis-transport", "empty-array", 0)

    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", original, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", logical_copy, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", changed_dtype, owner=owner
    ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", changed_shape, owner=owner
    ) is False
    assert calls == [original]


@pytest.mark.parametrize(
    "exception_expression",
    (
        "RuntimeError('ordinary')",
        "KeyboardInterrupt('interrupt')",
        "SystemExit('exit')",
    ),
)
def test_workflow_continuation_dispatcher_contains_recipient_baseexceptions_in_process(
    exception_expression,
):
    script = f"""
from PyQt5.QtCore import QCoreApplication, QObject
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import AnalysisTransportReady

app = QCoreApplication.instance() or QCoreApplication([])
bus = SequenceEventBus()
owner = QObject()
bus.register_workflow_continuation_lifecycle_owner(owner)

def hostile(_message):
    raise {exception_expression}

bus.register_workflow_continuation_recipient(
    "analysis-transport", "hostile", hostile
)
message = AnalysisTransportReady(
    "analysis", "source", "record", 0, {{"Label": "OK"}}
)
assert bus.deliver_workflow_continuation(
    ("analysis-transport", "analysis", 0),
    "analysis-transport",
    message,
    owner=owner,
) is False
print("process-alive")
"""
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "process-alive"


def test_workflow_continuation_dispatcher_acks_each_recipient_independently():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    calls = {"first": 0, "second": 0}
    failures_remaining = 10

    def first(_message):
        calls["first"] += 1
        return True

    def second(_message):
        nonlocal failures_remaining
        calls["second"] += 1
        if failures_remaining:
            failures_remaining -= 1
            raise SystemExit("retry this recipient")
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "first", first
    )
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "second", second
    )
    message = messages.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )
    delivery_id = ("analysis-transport", "analysis", 0)

    for _ in range(10):
        assert bus.deliver_workflow_continuation(
            delivery_id, "analysis-transport", message, owner=owner
        ) is False
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    ) is True
    assert bus.deliver_workflow_continuation(
        delivery_id, "analysis-transport", message, owner=owner
    ) is True

    assert calls == {"first": 1, "second": 11}
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 1


def test_destroyed_continuation_recipient_is_never_dereferenced():
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    lifecycle_owner = _registered_continuation_owner(bus)
    owner = QObject()
    calls = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "owned",
        lambda message: calls.append(message) or True,
        owner=owner,
    )
    sip.delete(owner)
    message = messages.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "OK"}
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 0),
        "analysis-transport",
        message,
        owner=lifecycle_owner,
    ) is False
    assert calls == []


def test_workflow_continuation_delivery_history_and_pending_are_bounded():
    completed_bus = SequenceEventBus()
    completed_owner = _registered_continuation_owner(completed_bus)
    completed_bus.register_workflow_continuation_recipient(
        "analysis-transport", "accept", lambda _message: True
    )
    for generation in range(256):
        message = messages.AnalysisTransportReady(
            "analysis", "source", "record", generation, {"Label": "OK"}
        )
        assert completed_bus.deliver_workflow_continuation(
            ("analysis-transport", "analysis", generation),
            "analysis-transport",
            message,
            owner=completed_owner,
        )
    assert completed_bus.completed_workflow_continuation_delivery_count == 128

    pending_bus = SequenceEventBus()
    pending_owner = _registered_continuation_owner(pending_bus)
    pending_bus.register_workflow_continuation_recipient(
        "analysis-transport", "blocked", lambda _message: False
    )
    for generation in range(256):
        message = messages.AnalysisTransportReady(
            "analysis", "source", "record", generation, {"Label": "OK"}
        )
        assert pending_bus.deliver_workflow_continuation(
            ("analysis-transport", "analysis", generation),
            "analysis-transport",
            message,
            owner=pending_owner,
        ) is False
    assert pending_bus.pending_workflow_continuation_delivery_count == 128

    recipient_bus = SequenceEventBus()
    for index in range(recipient_bus.workflow_continuation_recipient_limit):
        recipient_bus.register_workflow_continuation_recipient(
            "label-commit", f"recipient-{index}", lambda _message: True
        )
    with pytest.raises(RuntimeError, match="recipient registry is full"):
        recipient_bus.register_workflow_continuation_recipient(
            "label-commit", "overflow", lambda _message: True
        )


def test_stale_delivery_id_never_replays_into_next_workflow_generation():
    bus = SequenceEventBus()
    owner = _registered_continuation_owner(bus)
    observed_generations = []
    bus.register_workflow_continuation_recipient(
        "analysis-transport",
        "transport",
        lambda message: observed_generations.append(
            message.workflow_generation
        )
        or True,
    )
    first = messages.AnalysisTransportReady(
        "analysis", "source", "record", 0, {"Label": "first"}
    )
    second = messages.AnalysisTransportReady(
        "analysis", "source", "record", 1, {"Label": "second"}
    )

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 0),
        "analysis-transport",
        first,
        owner=owner,
    )
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 0),
        "analysis-transport",
        second,
        owner=owner,
    ) is False
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 1),
        "analysis-transport",
        second,
        owner=owner,
    )

    assert observed_generations == [0, 1]


@pytest.mark.parametrize("retirement", ("replacement", "owner-destroyed"))
def test_128_pending_deliveries_fail_closed_until_owner_explicitly_abandons(
    retirement,
):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    lifecycle_owner = QObject()
    other_owner = QObject()
    recipient_owner = QObject()
    replacement_owner = QObject()
    bus.register_workflow_continuation_lifecycle_owner(lifecycle_owner)
    bus.register_workflow_continuation_lifecycle_owner(other_owner)
    calls = {"old": 0, "new": 0}

    def old(_message):
        calls["old"] += 1
        return False

    def new(_message):
        calls["new"] += 1
        return True

    bus.register_workflow_continuation_recipient(
        "analysis-transport", "same-name", old, owner=recipient_owner
    )
    pending = []
    for generation in range(128):
        message = messages.AnalysisTransportReady(
            "analysis", "source", "record", generation, {"Label": "old"}
        )
        delivery_id = ("analysis-transport", "analysis", generation)
        pending.append((delivery_id, message))
        assert bus.deliver_workflow_continuation(
            delivery_id,
            "analysis-transport",
            message,
            owner=lifecycle_owner,
        ) is False

    assert bus.pending_workflow_continuation_delivery_count == 128
    assert bus.completed_workflow_continuation_delivery_count == 0
    if retirement == "replacement":
        bus.register_workflow_continuation_recipient(
            "analysis-transport",
            "same-name",
            new,
            owner=replacement_owner,
        )
    else:
        sip.delete(recipient_owner)
        bus.register_workflow_continuation_recipient(
            "analysis-transport",
            "same-name",
            new,
            owner=replacement_owner,
        )

    for delivery_id, message in pending:
        assert bus.deliver_workflow_continuation(
            delivery_id,
            "analysis-transport",
            message,
            owner=lifecycle_owner,
        ) is False
    assert bus.pending_workflow_continuation_delivery_count == 128
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert calls == {"old": 128, "new": 0}

    overflow = messages.AnalysisTransportReady(
        "analysis", "source", "record", 128, {"Label": "new"}
    )
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 128),
        "analysis-transport",
        overflow,
        owner=lifecycle_owner,
    ) is False
    assert calls["new"] == 0

    delivery_ids = tuple(delivery_id for delivery_id, _message in pending)
    assert bus.abandon_workflow_continuations(
        delivery_ids,
        owner=other_owner,
        reason="unrelated-workflow-disconnect",
    ) == 0
    assert bus.pending_workflow_continuation_delivery_count == 128
    assert bus.abandon_workflow_continuations(
        delivery_ids,
        owner=lifecycle_owner,
        reason="workflow-disconnect",
    ) == 128
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 128

    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 128),
        "analysis-transport",
        overflow,
        owner=lifecycle_owner,
    ) is True
    assert calls["new"] == 1


def test_owner_change_retires_only_exact_registration_without_acknowledging_pending():
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    owner = QObject()
    lifecycle_owner = _registered_continuation_owner(bus)
    calls = []

    def recipient(_message):
        calls.append(True)
        return False

    bus.register_workflow_continuation_recipient(
        "label-commit", "owned", recipient
    )
    bus.register_workflow_continuation_recipient(
        "label-commit", "owned", recipient, owner=owner
    )
    message = messages.CommitRecordingLabelRequested(
        "command", "record", "OK", ()
    )
    delivery_id = ("label-commit", "command", 0)
    assert bus.deliver_workflow_continuation(
        delivery_id, "label-commit", message, owner=lifecycle_owner
    ) is False
    assert calls == [True]

    sip.delete(owner)
    assert bus.deliver_workflow_continuation(
        delivery_id, "label-commit", message, owner=lifecycle_owner
    ) is False
    assert calls == [True]
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert bus.completed_workflow_continuation_delivery_count == 0


def test_explicit_abandonment_diagnostics_and_event_bus_disconnect_are_bounded():
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    lifecycle_owner = QObject()
    bus.register_workflow_continuation_lifecycle_owner(lifecycle_owner)
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "blocked", lambda _message: False
    )

    for generation in range(256):
        message = messages.AnalysisTransportReady(
            "analysis", "source", "record", generation, {"Label": "old"}
        )
        delivery_id = ("analysis-transport", "analysis", generation)
        assert bus.deliver_workflow_continuation(
            delivery_id,
            "analysis-transport",
            message,
            owner=lifecycle_owner,
        ) is False
        assert bus.abandon_workflow_continuations(
            (delivery_id,),
            owner=lifecycle_owner,
            reason="workflow-cancellation",
        ) == 1

    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 128
    assert all(
        diagnostic.reason == "workflow-cancellation"
        for diagnostic in bus.abandoned_workflow_continuation_diagnostics
    )

    final = messages.AnalysisTransportReady(
        "analysis", "source", "record", 256, {"Label": "old"}
    )
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", 256),
        "analysis-transport",
        final,
        owner=lifecycle_owner,
    ) is False
    bus.close_workflow_continuation_dispatcher()
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.completed_workflow_continuation_delivery_count == 0
    assert bus.abandoned_workflow_continuation_delivery_count == 128
    assert bus.abandoned_workflow_continuation_diagnostics[-1].reason == (
        "event-bus-disconnect"
    )


def test_lifecycle_owner_gc_and_destroyed_wrapper_never_reenter_dead_qobjects():
    script = """
import gc
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QObject
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import AnalysisTransportReady

app = QCoreApplication.instance() or QCoreApplication([])
for generation in range(256):
    bus = SequenceEventBus()
    lifecycle_owner = QObject()
    recipient_owner = QObject()
    bus.register_workflow_continuation_lifecycle_owner(lifecycle_owner)
    bus.register_workflow_continuation_recipient(
        "analysis-transport", "blocked", lambda _message: False,
        owner=recipient_owner,
    )
    message = AnalysisTransportReady(
        "analysis", "source", "record", generation, {"Label": "old"}
    )
    assert bus.deliver_workflow_continuation(
        ("analysis-transport", "analysis", generation),
        "analysis-transport",
        message,
        owner=lifecycle_owner,
    ) is False
    sip.delete(lifecycle_owner)
    assert bus.abandon_workflow_continuations(
        (("analysis-transport", "analysis", generation),),
        owner=lifecycle_owner,
        reason="destroyed-owner",
    ) == 0
    bus.close_workflow_continuation_dispatcher()
    del recipient_owner, lifecycle_owner, message, bus
    gc.collect()
print("gc-safe")
"""
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "gc-safe"


def test_lifecycle_owner_registry_is_globally_bounded_without_evicting_active():
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    owners = [
        QObject()
        for _index in range(bus.workflow_continuation_lifecycle_owner_limit)
    ]
    for owner in owners:
        bus.register_workflow_continuation_lifecycle_owner(owner)

    assert bus.workflow_continuation_lifecycle_owner_count == len(owners)
    with pytest.raises(RuntimeError, match="lifecycle owner registry is full"):
        bus.register_workflow_continuation_lifecycle_owner(QObject())
    assert bus.workflow_continuation_lifecycle_owner_count == len(owners)

    sip.delete(owners[0])
    replacement = QObject()
    bus.register_workflow_continuation_lifecycle_owner(replacement)
    assert bus.workflow_continuation_lifecycle_owner_count == len(owners)


def test_continuation_registry_has_one_global_bound_across_many_kinds():
    bus = SequenceEventBus()
    limit = bus.workflow_continuation_recipient_total_limit
    callbacks = []
    for index in range(limit):
        callback = lambda _message, index=index: index >= 0
        callbacks.append(callback)
        bus.register_workflow_continuation_recipient(
            f"kind-{index}", "recipient", callback
        )

    assert bus.workflow_continuation_recipient_count == limit
    assert bus.workflow_continuation_kind_count == limit
    for index in range(limit, 500):
        with pytest.raises(RuntimeError, match="recipient registry is full"):
            bus.register_workflow_continuation_recipient(
                f"kind-{index}", "recipient", lambda _message: True
            )
    assert bus.workflow_continuation_recipient_count == limit
    assert bus.workflow_continuation_kind_count == limit

    # The failed registration must not leave an empty outer kind, and active
    # registrations are never evicted to admit a newer one.
    assert "kind-499" not in bus._workflow_continuation_recipients
    for index, callback in enumerate(callbacks):
        bus.unregister_workflow_continuation_recipient(
            f"kind-{index}", "recipient", callback
        )
    assert bus.workflow_continuation_recipient_count == 0
    assert bus.workflow_continuation_kind_count == 0


@dataclass(frozen=True)
class NestedFrozenPayload:
    settings: dict


@dataclass(frozen=True)
class FrozenSessionPayload:
    nested: NestedFrozenPayload
    markers: list


@dataclass
class MutableDataclassPayload:
    values: list


class MutableCustomPayload:
    def __init__(self, values):
        self.values = values


class MutableText(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.values = []
        return instance


class TextFieldToken(StrEnum):
    VALUE = "token"


class IntegerFieldToken(IntEnum):
    ONE = 1


DATACLASS_TEXT_CALLS = []


@dataclass(init=False)
class DataclassText(str):
    def __str__(self):
        DATACLASS_TEXT_CALLS.append("called")
        return str.__str__(self)


@dataclass(frozen=True)
class DataclassIndex:
    calls: list

    def __index__(self):
        self.calls.append("called")
        return 1


@dataclass(frozen=True)
class DataclassChannelOrder:
    calls: list

    def __iter__(self):
        self.calls.append("called")
        return iter((0,))


@dataclass(init=False)
class DataclassArray(np.ndarray):
    pass


@dataclass(init=False)
class DataclassPath(PurePosixPath):
    pass


@dataclass(init=False)
class DataclassNumpyInteger(np.int64):
    pass


class MutableValueKind(Enum):
    SESSION = {"values": []}


class SemanticKind(Enum):
    SESSION = "session"


GLOBAL_ENUM_PROPERTY_STATE = []


class GlobalStatePropertyKind(Enum):
    SESSION = "session"

    @property
    def state(self):
        return GLOBAL_ENUM_PROPERTY_STATE


class MaliciousArray(np.ndarray):
    @property
    def ndim(self):
        raise AssertionError("untrusted ndim override was invoked")

    @property
    def shape(self):
        raise AssertionError("untrusted shape override was invoked")

    def tobytes(self, *_, **__):
        return bytearray(b"attacker-controlled")


class MaliciousFloat(np.float64):
    @property
    def dtype(self):
        raise AssertionError("untrusted scalar dtype override was invoked")


class NumpyIntegerKind(np.int64, Enum):
    ONE = 1


class HookedInt(int):
    calls = []

    def __index__(self):
        self.calls.append("index")
        return 1

    def __int__(self):
        self.calls.append("int")
        return 1

    def __bool__(self):
        self.calls.append("bool")
        return True

    def __str__(self):
        self.calls.append("str")
        return "1"


class HookedText(str):
    calls = []

    def __str__(self):
        self.calls.append("str")
        return "changed"


class HookedPath(PurePosixPath):
    calls = []

    @property
    def parts(self):
        self.calls.append("parts")
        return super().parts

    def __str__(self):
        self.calls.append("str")
        return super().__str__()


class HookedDict(dict):
    calls = []

    def items(self):
        self.calls.append("items")
        return super().items()


class HookedList(list):
    calls = []

    def __iter__(self):
        self.calls.append("iter")
        return super().__iter__()


class HookedTuple(tuple):
    calls = []

    def __iter__(self):
        self.calls.append("iter")
        return super().__iter__()


class HookedSet(set):
    calls = []

    def __iter__(self):
        self.calls.append("iter")
        return super().__iter__()


class HookedMapping(Mapping):
    calls = []

    def __getitem__(self, key):
        self.calls.append("getitem")
        return 1

    def __iter__(self):
        self.calls.append("iter")
        return iter(("key",))

    def __len__(self):
        self.calls.append("len")
        return 1


class CountingLookupKey:
    def __init__(self, value):
        self.value = value
        self.hash_calls = 0
        self.equality_calls = 0

    def __hash__(self):
        self.hash_calls += 1
        return self.value

    def __eq__(self, other):
        self.equality_calls += 1
        return type(other) is CountingLookupKey and self.value == other.value


class HookedOwnedCtypesArray(ctypes.Array):
    _type_ = ctypes.c_double
    _length_ = 4
    calls = []

    def __getattribute__(self, name):
        if name in ("_b_needsfree_", "_b_base_", "_objects"):
            type(self).calls.append(name)
        return object.__getattribute__(self, name)


HOSTILE_METACLASS_CALLS = []


class HostileType(EnumType):
    def __getattribute__(cls, name):
        HOSTILE_METACLASS_CALLS.append(name)
        return super().__getattribute__(name)


class HostilePlainType(type):
    def __getattribute__(cls, name):
        HOSTILE_METACLASS_CALLS.append(name)
        return super().__getattribute__(name)


class HostilePlainPayload(metaclass=HostilePlainType):
    pass


@dataclass(frozen=True)
class HostileDataclassPayload(metaclass=HostilePlainType):
    value: int = 1


class HostileNumpyEnum(np.int64, Enum, metaclass=HostileType):
    ONE = 1


class MutableTimezoneOffset(timedelta):
    calls = []

    @property
    def days(self):
        self.calls.append("days")
        raise AssertionError("timedelta subclass property was invoked")

    def total_seconds(self):
        self.calls.append("total_seconds")
        raise AssertionError("timedelta subclass method was invoked")


class MutableTimezoneName(str):
    calls = []

    def __str__(self):
        self.calls.append("str")
        raise AssertionError("string subclass hook was invoked")


class HostileTimezone(tzinfo):
    def __init__(self):
        self.calls = []

    def utcoffset(self, _value):
        self.calls.append("utcoffset")
        raise AssertionError("custom tzinfo hook was invoked")

    def dst(self, _value):
        self.calls.append("dst")
        raise AssertionError("custom tzinfo hook was invoked")

    def tzname(self, _value):
        self.calls.append("tzname")
        raise AssertionError("custom tzinfo hook was invoked")


@dataclass(frozen=True)
class SafeMethodAndConstantPayload:
    version = 1
    value: int = 2

    def doubled(self):
        return self.value * 2

    @property
    def label(self):
        return "safe"


DATACLASS_PROPERTY_GLOBAL_STATE = []


class MutableDataclassDescriptor:
    def __init__(self):
        self.registry = []

    def __get__(self, _instance, _owner):
        return self.registry


@dataclass(frozen=True)
class DescriptorStatePayload:
    descriptor = MutableDataclassDescriptor()
    value: int = 1


@dataclass(frozen=True)
class PropertyGlobalStatePayload:
    value: int = 1

    @property
    def exposed_state(self):
        return DATACLASS_PROPERTY_GLOBAL_STATE


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def valid_start_command():
    return StartTestRequested(
        command_id="command-1",
        source="manual",
        label="SN-001",
        skip_sn_regex_validation=False,
        configuration_generation=3,
    )


@pytest.mark.parametrize(
    "factory, value",
    [
        (
            lambda value: replace(valid_start_command(), command_id=value),
            TextFieldToken.VALUE,
        ),
        (
            lambda value: replace(valid_start_command(), label=value),
            TextFieldToken.VALUE,
        ),
        (
            lambda value: replace(
                valid_start_command(), configuration_generation=value
            ),
            IntegerFieldToken.ONE,
        ),
        (
            lambda value: AudioCompleted("session", value, 1),
            IntegerFieldToken.ONE,
        ),
        (
            lambda value: replace(
                valid_start_command(), skip_sn_regex_validation=value
            ),
            TextFieldToken.VALUE,
        ),
        (
            lambda value: replace(
                valid_start_command(), skip_sn_regex_validation=value
            ),
            IntegerFieldToken.ONE,
        ),
        (
            lambda value: AudioBatch(
                "session",
                0,
                0,
                1,
                np.zeros((1, 1), dtype=np.float32),
                (value,),
            ),
            IntegerFieldToken.ONE,
        ),
        (
            lambda value: AudioBatch(
                "session",
                0,
                0,
                1,
                np.zeros((1, 1), dtype=np.float32),
                value,
            ),
            TextFieldToken.VALUE,
        ),
    ],
    ids=[
        "identifier-str-enum",
        "text-str-enum",
        "generation-int-enum",
        "cursor-int-enum",
        "boolean-str-enum",
        "boolean-int-enum",
        "channel-index-int-enum",
        "channel-order-str-enum",
    ],
)
def test_direct_fields_reject_enum_mixes_before_coercion_or_iteration(factory, value):
    with pytest.raises(ValueError):
        factory(value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: replace(valid_start_command(), command_id=value),
        lambda value: replace(valid_start_command(), label=value),
    ],
    ids=["identifier", "text"],
)
def test_direct_text_fields_reject_dataclass_string_before_string_behavior(factory):
    DATACLASS_TEXT_CALLS.clear()
    value = DataclassText("text")

    with pytest.raises(ValueError):
        factory(value)

    assert DATACLASS_TEXT_CALLS == []


def test_direct_fields_reject_primitive_subclasses_without_conversion_hooks():
    HookedText.calls.clear()
    HookedInt.calls.clear()

    with pytest.raises(ValueError):
        replace(valid_start_command(), label=HookedText("SN"))
    with pytest.raises(ValueError):
        replace(valid_start_command(), configuration_generation=HookedInt(1))

    assert HookedText.calls == []
    assert HookedInt.calls == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: replace(
            valid_start_command(), configuration_generation=value
        ),
        lambda value: AudioCompleted("session", value, 1),
        lambda value: replace(
            valid_start_command(), skip_sn_regex_validation=value
        ),
        lambda value: AudioBatch(
            "session",
            0,
            0,
            1,
            np.zeros((1, 1), dtype=np.float32),
            (value,),
        ),
    ],
    ids=["generation", "cursor", "boolean", "channel-index"],
)
def test_direct_fields_reject_dataclass_index_without_calling_hook(factory):
    calls = []
    value = DataclassIndex(calls)

    with pytest.raises(ValueError):
        factory(value)

    assert calls == []


def test_channel_order_rejects_dataclass_iterable_without_calling_hook():
    calls = []
    value = DataclassChannelOrder(calls)

    with pytest.raises(ValueError):
        AudioBatch(
            "session",
            0,
            0,
            1,
            np.zeros((1, 1), dtype=np.float32),
            value,
        )

    assert calls == []


@pytest.mark.parametrize(
    "payload",
    [
        np.arange(3, dtype=np.float32).view(DataclassArray),
        DataclassPath("recordings/session.wav"),
        DataclassNumpyInteger(3),
    ],
    ids=["ndarray", "pure-path", "numpy-scalar"],
)
def test_dataclass_hybrids_reject_before_specialized_payload_dispatch(payload):
    with pytest.raises(TypeError, match="explicit snapshot"):
        ConfigurationChanged(1, {"payload": payload})


def test_numpy_scalar_enum_hybrid_is_rejected_before_numpy_normalization():
    with pytest.raises(TypeError, match="primitive stable token"):
        ConfigurationChanged(1, {"payload": NumpyIntegerKind.ONE})


@pytest.mark.parametrize(
    "payload, message_match",
    [
        (HostilePlainPayload(), "unsupported"),
        (HostileDataclassPayload(), "explicit snapshot"),
        (HostileNumpyEnum.ONE, "primitive stable token"),
    ],
    ids=["plain", "dataclass", "numpy-enum"],
)
def test_payload_type_classification_bypasses_hostile_metaclass_hooks(
    payload, message_match
):
    HOSTILE_METACLASS_CALLS.clear()

    with pytest.raises(TypeError, match=message_match):
        ConfigurationChanged(1, {"payload": payload})

    assert HOSTILE_METACLASS_CALLS == []


def test_messages_are_frozen_values():
    command = valid_start_command()

    with pytest.raises(FrozenInstanceError):
        command.label = "SN-002"

    assert not hasattr(command, "__dict__")
    with pytest.raises(AttributeError):
        object.__setattr__(command, "__dict__", {"label": "SN-003"})


def test_configuration_snapshot_defensively_copies_and_freezes_nested_values():
    sequence_config = {
        "mode": "record",
        "channels": [0, 1],
        "nested": {"limits": [1.0, 2.0]},
    }
    analysis_config = {"algorithms": [{"name": "rms"}]}
    mic = {"name": "input", "hostapi": {"index": 2}}
    streaming_reference = np.arange(4, dtype=np.float32)

    snapshot = ConfigurationSnapshot(
        sequence_config=sequence_config,
        analysis_config=analysis_config,
        mic=mic,
        speaker=None,
        mic_channels=(0, 1),
        using_config_path="configs/line-a.json",
        streaming_stimulus_data=streaming_reference,
    )
    sequence_config["mode"] = "import"
    sequence_config["channels"].append(9)
    analysis_config["algorithms"][0]["name"] = "changed"
    mic["hostapi"]["index"] = 99
    streaming_reference[:] = -1

    assert snapshot.sequence_config["mode"] == "record"
    assert snapshot.sequence_config["channels"] == (0, 1)
    assert snapshot.analysis_config["algorithms"][0]["name"] == "rms"
    assert snapshot.mic["hostapi"]["index"] == 2
    assert np.array_equal(snapshot.streaming_stimulus_data, np.arange(4, dtype=np.float32))
    assert snapshot.streaming_stimulus_data.flags.writeable is False
    with pytest.raises(ValueError):
        snapshot.streaming_stimulus_data.setflags(write=True)
    with pytest.raises(TypeError):
        snapshot.sequence_config["mode"] = "play"


def test_configuration_event_and_import_command_keep_frozen_snapshot():
    raw = {"mode": "record", "channels": [1, 2]}
    event = ConfigurationChanged(configuration_generation=4, configuration_snapshot=raw)
    command = LoadImportedAudioRequested(
        command_id="command-2",
        import_id="import-1",
        mode="audio",
        selected_path="record.wav",
        configuration_snapshot=raw,
    )
    raw["channels"].append(3)

    assert event.configuration_snapshot["channels"] == (1, 2)
    assert command.configuration_snapshot["channels"] == (1, 2)


def test_arbitrary_frozen_dataclass_payload_is_rejected():
    settings = {"channels": [0, 1], "limits": {"upper": [1.0, 2.0]}}
    markers = [{"name": "start"}]
    snapshot = FrozenSessionPayload(
        nested=NestedFrozenPayload(settings=settings),
        markers=markers,
    )

    with pytest.raises(TypeError, match="explicit snapshot"):
        BeginRecordingRequested(
            command_id="command-3",
            session_id="session-3",
            replay=False,
            session_snapshot=snapshot,
        )


def test_exact_configuration_snapshot_is_reused_after_recursive_freezing():
    source = {"channels": [0, 1]}
    samples = np.arange(4, dtype=np.float32)
    snapshot = ConfigurationSnapshot(
        sequence_config=source,
        analysis_config={"thresholds": [1.0, 2.0]},
        mic={"name": "input"},
        speaker=None,
        mic_channels=(0, 1),
        using_config_path="configs/line-a.json",
        streaming_stimulus_data=samples,
    )

    event = ConfigurationChanged(1, {"snapshot": snapshot})
    frozen = event.configuration_snapshot["snapshot"]

    assert type(frozen) is ConfigurationSnapshot
    assert frozen is snapshot
    assert frozen.sequence_config is snapshot.sequence_config
    assert frozen.sequence_config["channels"] == (0, 1)
    assert frozen.analysis_config["thresholds"] == (1.0, 2.0)
    assert frozen.mic_channels == (0, 1)
    assert np.array_equal(frozen.streaming_stimulus_data, samples)
    assert frozen.streaming_stimulus_data is snapshot.streaming_stimulus_data
    assert frozen.streaming_stimulus_data.flags.writeable is False


def test_configuration_snapshot_public_constructor_runs_validation():
    with pytest.raises(ValueError, match="ordered collection"):
        ConfigurationSnapshot({}, {}, mic_channels={0, 1})


def test_configuration_snapshot_constructor_freezes_raw_fields_once(monkeypatch):
    leaf = 987_654_321
    leaf_visits = []
    original_freeze = messages._freeze_payload

    def track_freeze(value, active_path=None):
        if value is leaf:
            leaf_visits.append(value)
        return original_freeze(value, active_path)

    monkeypatch.setattr(messages, "_freeze_payload", track_freeze)

    snapshot = ConfigurationSnapshot(
        sequence_config={"nested": {"leaf": leaf}},
        analysis_config={},
    )
    rebuilt = ConfigurationChanged(1, {"snapshot": snapshot}).configuration_snapshot[
        "snapshot"
    ]

    assert rebuilt is snapshot
    assert rebuilt.sequence_config["nested"]["leaf"] == leaf
    assert leaf_visits == [leaf]


def test_exact_configuration_snapshot_rebuild_reuses_frozen_nested_leaves(
    monkeypatch,
):
    samples = np.arange(6, dtype=np.float32).reshape(3, 2)
    snapshot = ConfigurationSnapshot(
        sequence_config={"nested": {"channels": [0, 1]}},
        analysis_config={"thresholds": [1.0, 2.0]},
        streaming_stimulus_data=samples,
    )
    copies = []
    original_copy = messages._copy_numpy_array_to_immutable

    def track_copy(value, *, validate_provenance):
        copies.append(value)
        return original_copy(value, validate_provenance=validate_provenance)

    monkeypatch.setattr(messages, "_copy_numpy_array_to_immutable", track_copy)

    rebuilt = ConfigurationChanged(1, {"snapshot": snapshot}).configuration_snapshot[
        "snapshot"
    ]

    assert rebuilt is snapshot
    assert rebuilt.sequence_config is snapshot.sequence_config
    assert rebuilt.sequence_config["nested"] is snapshot.sequence_config["nested"]
    assert rebuilt.analysis_config is snapshot.analysis_config
    assert rebuilt.streaming_stimulus_data is snapshot.streaming_stimulus_data
    assert copies == []


def test_payload_freezing_uses_no_module_global_traversal_context():
    assert not hasattr(messages, "_ACTIVE_PAYLOAD_PATH")


@pytest.mark.parametrize(
    "message_type",
    [StartTestRequested, AudioBatch, ConfigurationSnapshot],
    ids=["command", "audio", "snapshot"],
)
def test_public_message_classes_reject_hostile_subclass_definition(message_type):
    calls = []

    class HostileDescriptor:
        def __get__(self, _instance, _owner):
            calls.append("descriptor")
            raise AssertionError("hostile descriptor was invoked")

    with pytest.raises(TypeError, match="sealed"):
        class HostileMessage(message_type):
            hostile = HostileDescriptor()

            def __getattribute__(self, name):
                calls.append(name)
                raise AssertionError("hostile instance hook was invoked")

    assert calls == []


def test_audio_batch_copies_callback_memory_and_exposes_read_only_array():
    source = np.arange(8, dtype=np.float32).reshape(4, 2)
    batch = AudioBatch.from_callback(
        session_id="s1",
        sequence_no=0,
        sample_start=0,
        multi=source,
        channel_order=(0, 1),
    )
    source[:] = -1

    assert batch.sample_stop == 4
    assert np.array_equal(batch.multi, np.arange(8, dtype=np.float32).reshape(4, 2))
    assert batch.multi.flags.c_contiguous is True
    assert batch.multi.flags.writeable is False
    with pytest.raises(ValueError):
        batch.multi.setflags(write=True)


def test_audio_batch_copies_optional_mono_callback_memory():
    multi = np.arange(6, dtype=np.float32).reshape(3, 2)
    mono = np.arange(3, dtype=np.float32)
    batch = AudioBatch.from_callback(
        session_id="s1",
        sequence_no=1,
        sample_start=4,
        multi=multi,
        mono=mono,
        channel_order=(0, 1),
    )
    mono[:] = -1

    assert np.array_equal(batch.mono, np.arange(3, dtype=np.float32))
    assert batch.mono.flags.c_contiguous is True
    assert batch.mono.flags.writeable is False
    with pytest.raises(ValueError):
        batch.mono.setflags(write=True)


def test_canonical_audio_batch_preserves_narrow_legacy_payload_indexing():
    batch = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=np.array([[1.0], [2.0]], dtype=np.float32),
        mono=np.array([1.0, 2.0], dtype=np.float32),
        channel_order=(0,),
    )

    assert batch["multi"] is batch.multi
    assert batch["mono"] is batch.mono
    with pytest.raises(KeyError):
        batch["other"]


def test_audio_batch_callback_retains_one_direct_copy_per_array(monkeypatch):
    multi = np.arange(6, dtype=np.float32).reshape(3, 2)
    mono = np.arange(3, dtype=np.float32)
    numpy_array_calls = []
    original_array = messages.np.array

    def track_numpy_array(*args, **kwargs):
        numpy_array_calls.append((args, kwargs))
        return original_array(*args, **kwargs)

    callback_copy_calls = []
    original_callback_copy = messages._freeze_callback_numpy_array

    def track_callback_copy(value):
        callback_copy_calls.append(value)
        return original_callback_copy(value)

    monkeypatch.setattr(messages.np, "array", track_numpy_array)
    monkeypatch.setattr(messages, "_freeze_callback_numpy_array", track_callback_copy)

    batch = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=multi,
        mono=mono,
        channel_order=(0, 1),
    )

    assert callback_copy_calls == [multi, mono]
    assert numpy_array_calls == []
    assert type(batch.multi.base) is bytes
    assert type(batch.mono.base) is bytes
    assert not np.shares_memory(batch.multi, multi)
    assert not np.shares_memory(batch.mono, mono)


def test_audio_batch_constructor_reuses_valid_immutable_payload_arrays():
    first = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=np.arange(6, dtype=np.float32).reshape(3, 2),
        mono=np.arange(3, dtype=np.float32),
        channel_order=(0, 1),
    )

    second = AudioBatch(
        session_id="session",
        sequence_no=1,
        sample_start=3,
        sample_stop=6,
        multi=first.multi,
        mono=first.mono,
        channel_order=(0, 1),
    )

    assert second.multi is first.multi
    assert second.mono is first.mono


@pytest.mark.parametrize("constructor", ["direct", "callback"])
@pytest.mark.parametrize(
    "mono_shape",
    [(3, 0), (3, 2), (3, 2, 1), (2,), (2, 1)],
    ids=[
        "zero-channels",
        "multiple-channels",
        "three-dimensional",
        "short-1d",
        "short-2d",
    ],
)
def test_audio_batch_rejects_invalid_mono_shape_before_retained_copy(
    constructor, mono_shape, monkeypatch
):
    copies = []
    original_direct_freeze = messages._freeze_numpy_array
    original_callback_freeze = messages._freeze_callback_numpy_array

    def track_direct_freeze(value, *, reuse_immutable=False):
        copies.append(value)
        return original_direct_freeze(value, reuse_immutable=reuse_immutable)

    def track_callback_freeze(value):
        copies.append(value)
        return original_callback_freeze(value)

    monkeypatch.setattr(messages, "_freeze_numpy_array", track_direct_freeze)
    monkeypatch.setattr(
        messages,
        "_freeze_callback_numpy_array",
        track_callback_freeze,
    )
    arguments = {
        "session_id": "session",
        "sequence_no": 0,
        "sample_start": 0,
        "multi": np.zeros((3, 1), dtype=np.float32),
        "mono": np.zeros(mono_shape, dtype=np.float32),
        "channel_order": (0,),
    }

    with pytest.raises(ValueError, match="shape"):
        if constructor == "callback":
            AudioBatch.from_callback(**arguments)
        else:
            AudioBatch(sample_stop=3, **arguments)

    assert copies == []


@pytest.mark.parametrize("constructor", ["direct", "callback"])
@pytest.mark.parametrize(
    "mono_shape",
    [(3,), (3, 1)],
    ids=["one-dimensional", "column"],
)
def test_audio_batch_accepts_exact_mono_shapes(constructor, mono_shape):
    arguments = {
        "session_id": "session",
        "sequence_no": 0,
        "sample_start": 0,
        "multi": np.zeros((3, 1), dtype=np.float32),
        "mono": np.zeros(mono_shape, dtype=np.float32),
        "channel_order": (0,),
    }

    if constructor == "callback":
        batch = AudioBatch.from_callback(**arguments)
    else:
        batch = AudioBatch(sample_stop=3, **arguments)

    assert batch.mono.shape == mono_shape


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("session_id", HookedText("session")),
        ("sequence_no", HookedInt(0)),
        ("sample_start", HookedInt(0)),
        ("channel_order", (HookedInt(0),)),
    ],
)
def test_audio_callback_rejects_behavioral_fields_before_retaining_copy(
    field_name, invalid_value, monkeypatch
):
    HookedText.calls.clear()
    HookedInt.calls.clear()
    copies = []
    original_copy = messages._freeze_callback_numpy_array

    def track_copy(value):
        copies.append(value)
        return original_copy(value)

    monkeypatch.setattr(messages, "_freeze_callback_numpy_array", track_copy)
    arguments = {
        "session_id": "session",
        "sequence_no": 0,
        "sample_start": 0,
        "multi": np.zeros((1, 1), dtype=np.float32),
        "channel_order": (0,),
    }
    arguments[field_name] = invalid_value

    with pytest.raises(ValueError):
        AudioBatch.from_callback(**arguments)

    assert copies == []
    assert HookedText.calls == []
    assert HookedInt.calls == []


def test_audio_callback_rejects_ndarray_subclass_before_copy_or_hooks(monkeypatch):
    source = np.zeros((1, 1), dtype=np.float32).view(MaliciousArray)
    copies = []
    monkeypatch.setattr(
        messages,
        "_freeze_callback_numpy_array",
        lambda value: copies.append(value),
    )

    with pytest.raises(TypeError, match="exact NumPy ndarray"):
        AudioBatch.from_callback(
            session_id="session",
            sequence_no=0,
            sample_start=0,
            multi=source,
            channel_order=(0,),
        )

    assert copies == []


@pytest.mark.parametrize(
    "multi, channel_order",
    [
        (np.zeros(1, dtype=np.float32), (0,)),
        (np.zeros((1, 2), dtype=np.float32), (0,)),
    ],
    ids=["dimensions", "channel-shape"],
)
def test_audio_callback_validates_shape_before_retaining_copy(
    multi, channel_order, monkeypatch
):
    copies = []
    monkeypatch.setattr(
        messages,
        "_freeze_callback_numpy_array",
        lambda value: copies.append(value),
    )

    with pytest.raises(ValueError):
        AudioBatch.from_callback(
            session_id="session",
            sequence_no=0,
            sample_start=0,
            multi=multi,
            channel_order=channel_order,
        )

    assert copies == []


def test_audio_callback_validates_optional_mono_shape_before_any_retained_copy(
    monkeypatch,
):
    copies = []
    monkeypatch.setattr(
        messages,
        "_freeze_callback_numpy_array",
        lambda value: copies.append(value),
    )

    with pytest.raises(ValueError, match="same number of frames"):
        AudioBatch.from_callback(
            session_id="session",
            sequence_no=0,
            sample_start=0,
            multi=np.zeros((2, 1), dtype=np.float32),
            mono=np.zeros(1, dtype=np.float32),
            channel_order=(0,),
        )

    assert copies == []


@pytest.mark.parametrize(
    "selection, expected_order",
    [([], (0, 1, 2)), ([1], (1,)), ([2, 0], (2, 0))],
    ids=["all-channels", "selected-channel", "reordered-channels"],
)
def test_audio_batch_callback_factory_copies_real_sounddevice_cffi_arrays(
    selection, expected_order, monkeypatch
):
    sounddevice = pytest.importorskip("sounddevice")
    from base.streaming_audio_processor import StreamingAudioProcessor

    values = np.arange(12, dtype=np.float32)
    owner = sounddevice._ffi.new("float[]", values.tolist())
    callback_buffer = sounddevice._ffi.buffer(owner, values.nbytes)
    indata = sounddevice._array(callback_buffer, 3, "float32")
    multi = StreamingAudioProcessor._select_multi(indata, selection)
    callback_copies = []
    original_freeze = messages._freeze_callback_numpy_array

    def record_callback_copy(value):
        callback_copies.append(value)
        return original_freeze(value)

    monkeypatch.setattr(messages, "_freeze_callback_numpy_array", record_callback_copy)

    batch = AudioBatch.from_callback(
        session_id="sounddevice-session",
        sequence_no=0,
        sample_start=0,
        multi=multi,
        mono=multi[:, 0],
        channel_order=expected_order,
    )
    indata[:] = -1

    expected_multi = values.reshape(4, 3)
    if selection:
        expected_multi = expected_multi[:, selection]
    assert np.array_equal(batch.multi, expected_multi)
    assert np.array_equal(batch.mono, expected_multi[:, 0])
    assert batch.multi.flags.c_contiguous is True
    assert batch.multi.flags.writeable is False
    assert batch.channel_order == expected_order
    assert len(callback_copies) == 2
    assert all(type(value) is np.ndarray for value in callback_copies)
    assert type(batch.multi.base) is bytes
    assert type(batch.mono.base) is bytes


def test_recording_batch_ready_owns_read_only_ui_memory():
    display = np.arange(6, dtype=np.float32).reshape(3, 2)
    event = RecordingBatchReady(
        session_id="session-1",
        sequence_no=0,
        sample_start=0,
        sample_stop=3,
        display=display,
    )
    display[:] = -1

    assert np.array_equal(event.display, np.arange(6, dtype=np.float32).reshape(3, 2))
    assert event.display.flags.writeable is False
    with pytest.raises(ValueError):
        event.display.setflags(write=True)


def audio_batch_with_sequence_no(value):
    return AudioBatch(
        session_id="session-1",
        sequence_no=value,
        sample_start=0,
        sample_stop=1,
        multi=np.zeros((1, 1), dtype=np.float32),
        channel_order=(0,),
    )


def audio_batch_with_sample_start(value):
    return AudioBatch(
        session_id="session-1",
        sequence_no=0,
        sample_start=value,
        sample_stop=2,
        multi=np.zeros((1, 1), dtype=np.float32),
        channel_order=(0,),
    )


def audio_batch_with_sample_stop(value):
    return AudioBatch(
        session_id="session-1",
        sequence_no=0,
        sample_start=0,
        sample_stop=value,
        multi=np.zeros((1, 1), dtype=np.float32),
        channel_order=(0,),
    )


def audio_completed_with_last_sequence_no(value):
    return AudioCompleted(session_id="session-1", last_sequence_no=value, sample_count=1)


def audio_completed_with_sample_count(value):
    return AudioCompleted(session_id="session-1", last_sequence_no=0, sample_count=value)


def audio_failed_with_last_sequence_no(value):
    return AudioFailed(
        session_id="session-1",
        last_sequence_no=value,
        error_code="device-error",
        message="failed",
    )


def audio_cancelled_with_last_sequence_no(value):
    return AudioCancelled(session_id="session-1", last_sequence_no=value, reason="cancel")


def recording_batch_with_sequence_no(value):
    return RecordingBatchReady(
        session_id="session-1",
        sequence_no=value,
        sample_start=0,
        sample_stop=1,
        display=np.zeros(1, dtype=np.float32),
    )


def recording_batch_with_sample_start(value):
    return RecordingBatchReady(
        session_id="session-1",
        sequence_no=0,
        sample_start=value,
        sample_stop=2,
        display=np.zeros(1, dtype=np.float32),
    )


def recording_batch_with_sample_stop(value):
    return RecordingBatchReady(
        session_id="session-1",
        sequence_no=0,
        sample_start=0,
        sample_stop=value,
        display=np.zeros(1, dtype=np.float32),
    )


def recording_completed_with_sample_count(value):
    return RecordingCompleted(
        session_id="session-1",
        sample_count=value,
        result_snapshot={"path": "record.wav"},
    )


@pytest.mark.parametrize(
    "factory",
    [
        audio_batch_with_sequence_no,
        audio_batch_with_sample_start,
        audio_batch_with_sample_stop,
        audio_completed_with_last_sequence_no,
        audio_completed_with_sample_count,
        audio_failed_with_last_sequence_no,
        audio_cancelled_with_last_sequence_no,
        recording_batch_with_sequence_no,
        recording_batch_with_sample_start,
        recording_batch_with_sample_stop,
        recording_completed_with_sample_count,
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 1.0, np.nan])
def test_audio_and_display_cursors_require_exact_integers(factory, invalid_value):
    with pytest.raises(ValueError):
        factory(invalid_value)


@pytest.mark.parametrize(
    "channel_order, channel_count",
    [
        ((-1,), 1),
        ((0, 0), 2),
        ((0.0,), 1),
        ((True,), 1),
        ((np.nan,), 1),
        ((), 0),
        ((0,), 2),
    ],
)
def test_audio_batch_rejects_invalid_channel_indices_and_orders(channel_order, channel_count):
    with pytest.raises(ValueError):
        AudioBatch(
            session_id="session-1",
            sequence_no=0,
            sample_start=0,
            sample_stop=1,
            multi=np.zeros((1, channel_count), dtype=np.float32),
            channel_order=channel_order,
        )


@pytest.mark.parametrize(
    "unordered_channels",
    [
        {0, 1},
        frozenset({0, 1}),
        {0: "left", 1: "right"},
        np.array([0, 1], dtype=np.int64),
    ],
    ids=["set", "frozenset", "mapping", "ndarray"],
)
def test_channel_order_rejects_unordered_collections(unordered_channels):
    with pytest.raises(ValueError, match="ordered collection"):
        AudioBatch(
            session_id="session-1",
            sequence_no=0,
            sample_start=0,
            sample_stop=1,
            multi=np.zeros((1, 2), dtype=np.float32),
            channel_order=unordered_channels,
        )


@pytest.mark.parametrize(
    "ordered_channels",
    [
        (1, 0),
        [1, 0],
    ],
    ids=["tuple", "list"],
)
def test_channel_order_preserves_ordered_inputs(ordered_channels):
    batch = AudioBatch(
        session_id="session-1",
        sequence_no=0,
        sample_start=0,
        sample_stop=1,
        multi=np.zeros((1, 2), dtype=np.float32),
        channel_order=ordered_channels,
    )

    assert batch.channel_order == (1, 0)


def test_configuration_snapshot_rejects_ndarray_mic_channel_order():
    with pytest.raises(ValueError, match="ordered collection"):
        ConfigurationSnapshot(
            sequence_config={},
            analysis_config={},
            mic_channels=np.array([0, 1], dtype=np.int64),
        )


def test_channel_order_rejects_as_strided_ndarray_before_element_read(monkeypatch):
    source = np.array([0], dtype=np.int64)
    hostile = np.lib.stride_tricks.as_strided(source, shape=(2,), strides=(0,))
    element_reads = []

    def record_element_read(*_args, **_kwargs):
        element_reads.append("read")
        raise AssertionError("channel ndarray element was read")

    monkeypatch.setattr(messages, "_exact_integer", record_element_read)

    with pytest.raises(ValueError, match="ordered collection"):
        messages._channel_order("channel_order", hostile, allow_empty=False)

    assert element_reads == []


def test_channel_order_rejects_generator_before_iteration():
    calls = []

    def channels():
        calls.append("iterated")
        yield 0

    with pytest.raises(ValueError, match="ordered collection"):
        AudioBatch(
            session_id="session-1",
            sequence_no=0,
            sample_start=0,
            sample_stop=1,
            multi=np.zeros((1, 1), dtype=np.float32),
            channel_order=channels(),
        )

    assert calls == []


def test_channel_order_rejects_container_subclass_before_iteration():
    channels = HookedList((0,))
    HookedList.calls.clear()

    with pytest.raises(ValueError, match="ordered collection"):
        AudioBatch(
            session_id="session-1",
            sequence_no=0,
            sample_start=0,
            sample_stop=1,
            multi=np.zeros((1, 1), dtype=np.float32),
            channel_order=channels,
        )

    assert HookedList.calls == []


@pytest.mark.parametrize(
    "mic_channels",
    [(-1,), (0, 0), (0.0,), (True,), (np.nan,)],
)
def test_configuration_snapshot_rejects_invalid_mic_channel_order(mic_channels):
    with pytest.raises(ValueError):
        ConfigurationSnapshot(
            sequence_config={},
            analysis_config={},
            mic_channels=mic_channels,
        )


def test_exact_integer_types_and_terminal_minus_one_are_accepted():
    batch = AudioBatch(
        session_id="session-1",
        sequence_no=np.int64(0),
        sample_start=np.int64(0),
        sample_stop=np.int64(1),
        multi=np.zeros((1, 2), dtype=np.float32),
        channel_order=(np.int64(2), np.int64(0)),
    )
    snapshot = ConfigurationSnapshot(
        sequence_config={},
        analysis_config={},
        mic_channels=(np.int64(2), np.int64(0)),
    )

    assert batch.sequence_no == 0
    assert batch.sample_start == 0
    assert batch.sample_stop == 1
    assert batch.channel_order == (2, 0)
    assert snapshot.mic_channels == (2, 0)
    assert AudioCompleted("session-1", -1, 0).last_sequence_no == -1
    assert AudioFailed("session-1", -1, "empty", "failed").last_sequence_no == -1
    assert AudioCancelled("session-1", -1, "cancelled").last_sequence_no == -1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AudioCompleted(session_id="", last_sequence_no=0, sample_count=4),
        lambda: ExportCompleted(
            job_id="", attempt_id="attempt-1", record_id="record-1", target_results=()
        ),
        lambda: ExportCompleted(
            job_id="job-1", attempt_id="", record_id="record-1", target_results=()
        ),
        lambda: RetryExportRequested(job_id="job-1", attempt_id=""),
        lambda: ShutdownFlushFailed(
            shutdown_generation=1,
            job_id="job-1",
            attempt_id="",
            failures=(),
        ),
    ],
)
def test_session_job_and_attempt_identifiers_must_be_non_empty(factory):
    with pytest.raises(ValueError):
        factory()


def test_event_bus_instances_own_independent_named_channels(qapp):
    first = SequenceEventBus()
    second = SequenceEventBus()
    first_messages = []
    second_messages = []
    first.commands.start_test_requested.connect(first_messages.append)
    second.commands.start_test_requested.connect(second_messages.append)

    command = valid_start_command()
    first.commands.start_test_requested.emit(command)

    assert first.commands is not second.commands
    assert first.events is not second.events
    assert first.commands.parent() is first
    assert first.events.parent() is first
    assert first_messages == [command]
    assert second_messages == []
    assert hasattr(first.commands, "begin_recording_requested")
    assert hasattr(first.events, "recording_completed")
    assert not hasattr(first.commands, "message")
    assert not hasattr(first.events, "message")


def test_queued_signal_cannot_observe_message_mutation_before_delivery(qapp):
    from PyQt5.QtCore import Qt

    bus = SequenceEventBus()
    received = []
    bus.commands.start_test_requested.connect(received.append, Qt.QueuedConnection)
    command = StartTestRequested(
        command_id=np.str_("command-queued"),
        source="manual",
        label=np.str_("SN-QUEUED"),
        skip_sn_regex_validation=np.bool_(False),
        configuration_generation=np.int64(4),
    )

    bus.commands.start_test_requested.emit(command)
    assert received == []
    with pytest.raises(FrozenInstanceError):
        command.label = "changed"
    with pytest.raises(AttributeError):
        object.__setattr__(command, "__dict__", {"label": "changed"})

    qapp.processEvents()

    assert received == [command]
    assert received[0].command_id == "command-queued"
    assert received[0].label == "SN-QUEUED"
    assert received[0].configuration_generation == 4
    assert received[0].skip_sn_regex_validation is False


def test_numeric_array_payload_is_alias_isolated_bytes_backed_and_read_only():
    source = np.arange(6, dtype=np.float32).reshape(3, 2)
    expected = source.copy()
    payload = {
        "path": Path("recordings/session.wav"),
        "kind": "session",
        "samples": source,
    }

    command = BeginRecordingRequested(
        command_id="command-array",
        session_id="session-array",
        replay=False,
        session_snapshot=payload,
    )
    source[:] = -1

    frozen = command.session_snapshot
    assert frozen["path"] == Path("recordings/session.wav")
    assert frozen["kind"] == "session"
    assert frozen["samples"].shape == (3, 2)
    assert frozen["samples"].dtype == np.dtype("float32")
    assert frozen["samples"].flags.c_contiguous is True
    assert np.array_equal(frozen["samples"], expected)
    assert type(frozen["samples"].base) is bytes
    assert frozen["samples"].flags.writeable is False
    with pytest.raises(AttributeError):
        frozen["samples"].mutable_state = []
    with pytest.raises(ValueError):
        frozen["samples"].setflags(write=True)


def test_ndarray_subclasses_are_rejected_without_calling_overrides():
    source = np.arange(6, dtype=np.float32).reshape(2, 3).view(MaliciousArray)

    with pytest.raises(TypeError, match="exact NumPy ndarray"):
        ConfigurationChanged(1, {"array": source})


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda source: np.lib.stride_tricks.as_strided(
            source[-1:], shape=(2,), strides=(source.itemsize,)
        ),
        lambda source: np.lib.stride_tricks.as_strided(
            source[:1], shape=(2,), strides=(-source.itemsize,)
        ),
    ],
    ids=["positive-stride-overrun", "negative-stride-underrun"],
)
def test_numpy_payload_rejects_stride_trick_views_outside_backing_allocation(
    view_factory,
):
    source = np.arange(4, dtype=np.float64)
    hostile_view = view_factory(source)

    with pytest.raises(TypeError, match="backing allocation"):
        ConfigurationChanged(1, {"array": hostile_view})


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda source: source[:, 0],
        lambda source: source[1:4],
        lambda source: source[::-1],
    ],
    ids=["column", "contiguous-subview", "reversed-strided"],
)
def test_recording_batch_accepts_views_of_module_immutable_arrays_with_one_copy(
    view_factory, monkeypatch
):
    batch = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=np.arange(12, dtype=np.float32).reshape(6, 2),
        channel_order=(0, 1),
    )
    source_view = view_factory(batch.multi)
    expected = np.array(source_view, copy=True, order="C")
    copies = []
    original_copy = messages._copy_numpy_array_to_immutable

    def track_copy(value, *, validate_provenance):
        copies.append(value)
        return original_copy(value, validate_provenance=validate_provenance)

    monkeypatch.setattr(messages, "_copy_numpy_array_to_immutable", track_copy)

    event = RecordingBatchReady(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        sample_stop=source_view.shape[0],
        display=source_view,
    )

    assert copies == [source_view]
    assert event.display is not source_view
    assert np.array_equal(event.display, expected)
    assert event.display.flags.c_contiguous is True
    assert event.display.flags.writeable is False
    assert type(event.display.base) is bytes


def test_recording_batch_reuses_direct_module_immutable_array_root(monkeypatch):
    batch = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=np.arange(6, dtype=np.float32).reshape(3, 2),
        channel_order=(0, 1),
    )
    copies = []
    monkeypatch.setattr(
        messages,
        "_copy_numpy_array_to_immutable",
        lambda value, *, validate_provenance: copies.append(value),
    )

    event = RecordingBatchReady("session", 0, 0, 3, batch.multi)

    assert event.display is batch.multi
    assert copies == []


def test_numpy_payload_rejects_out_of_bounds_view_of_module_immutable_array():
    batch = AudioBatch.from_callback(
        session_id="session",
        sequence_no=0,
        sample_start=0,
        multi=np.arange(6, dtype=np.float32).reshape(3, 2),
        channel_order=(0, 1),
    )
    hostile_view = np.lib.stride_tricks.as_strided(
        batch.multi[-1:, -1:],
        shape=(2,),
        strides=(batch.multi.itemsize,),
    )

    with pytest.raises(TypeError, match="backing allocation"):
        RecordingBatchReady("session", 0, 0, 2, hostile_view)


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda source: source,
        lambda source: source[1:5:2],
        lambda source: source[::-1],
        lambda source: source.reshape(2, 3).T,
    ],
    ids=["contiguous", "slice", "reversed", "transpose"],
)
def test_numpy_payload_preserves_valid_views_with_proven_allocation_bounds(
    view_factory,
):
    source = np.arange(6, dtype=np.float64)
    source_view = view_factory(source)
    expected = np.array(source_view, copy=True, order="C")

    frozen = ConfigurationChanged(1, {"array": source_view}).configuration_snapshot[
        "array"
    ]

    assert np.array_equal(frozen, expected)
    assert frozen.flags.c_contiguous is True
    assert frozen.flags.writeable is False


def test_numpy_payload_preserves_fixed_owned_ctypes_array_buffer():
    storage = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    source = np.ctypeslib.as_array(storage)

    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot[
        "array"
    ]

    assert np.array_equal(frozen, np.array([1.0, 2.0, 3.0, 4.0]))
    assert frozen.flags.writeable is False


def test_owned_ctypes_array_subclass_provenance_bypasses_attribute_hooks():
    storage = HookedOwnedCtypesArray(1.0, 2.0, 3.0, 4.0)
    source = np.ctypeslib.as_array(storage)
    HookedOwnedCtypesArray.calls.clear()

    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot[
        "array"
    ]

    assert np.array_equal(frozen, np.array([1.0, 2.0, 3.0, 4.0]))
    assert frozen.flags.writeable is False
    assert HookedOwnedCtypesArray.calls == []


@pytest.mark.parametrize(
    "buffer",
    [bytes(range(8)), bytearray(range(8))],
    ids=["bytes", "bytearray"],
)
def test_numpy_payload_preserves_concrete_sized_python_buffers(buffer):
    source = np.frombuffer(buffer, dtype=np.uint8)

    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot[
        "array"
    ]

    assert np.array_equal(frozen, np.arange(8, dtype=np.uint8))


def test_numpy_payload_preserves_offset_memoryview_extent():
    storage = bytearray(range(16))
    source = np.frombuffer(memoryview(storage)[4:12], dtype=np.uint8)

    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot[
        "array"
    ]

    assert np.array_equal(frozen, np.arange(4, 12, dtype=np.uint8))


def test_numpy_payload_rejects_oversized_pointer_derived_ctypes_buffer():
    storage = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    pointer = ctypes.cast(storage, ctypes.POINTER(ctypes.c_double))
    foreign = np.ctypeslib.as_array(pointer, shape=(5,))

    with pytest.raises(TypeError, match="backing allocation cannot be proven"):
        ConfigurationChanged(1, {"array": foreign})


def test_numpy_payload_rejects_foreign_pointer_provenance_through_base_chain():
    storage = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    pointer = ctypes.cast(storage, ctypes.POINTER(ctypes.c_double))
    foreign_view = np.ctypeslib.as_array(pointer, shape=(5,))[::-1]

    with pytest.raises(TypeError, match="backing allocation cannot be proven"):
        ConfigurationChanged(1, {"array": foreign_view})


def test_numpy_payload_rejects_forged_ctypes_array_extent():
    storage = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    forged_storage = (ctypes.c_double * 5).from_address(ctypes.addressof(storage))
    foreign = np.ctypeslib.as_array(forged_storage)

    with pytest.raises(TypeError, match="backing allocation cannot be proven"):
        ConfigurationChanged(1, {"array": foreign})


def test_audio_callback_factory_rejects_array_subclass_without_metadata_hooks():
    source = np.arange(6, dtype=np.float32).reshape(2, 3).view(MaliciousArray)

    with pytest.raises(TypeError, match="exact NumPy ndarray"):
        AudioBatch.from_callback(
            session_id="session",
            sequence_no=0,
            sample_start=5,
            multi=source,
            channel_order=(0, 1, 2),
        )


def test_object_dtype_array_is_rejected_before_nested_alias_can_cross_boundary():
    aliased = {"values": [1, 2]}
    object_array = np.array([aliased], dtype=object)

    with pytest.raises(TypeError):
        ConfigurationChanged(
            configuration_generation=1,
            configuration_snapshot={"unsafe": object_array},
        )


@pytest.mark.parametrize(
    "payload",
    [
        MutableDataclassPayload(values=[1, 2]),
        MutableCustomPayload(values=[1, 2]),
    ],
)
def test_mutable_or_unsupported_custom_payload_is_rejected(payload):
    with pytest.raises(TypeError):
        BeginRecordingRequested(
            command_id="command-mutable",
            session_id="session-mutable",
            replay=False,
            session_snapshot=payload,
        )


def identifier_messages():
    return [
        messages.StartTestRequested("command", "manual", "SN", False, 0),
        messages.ReplayRequested("command", "manual", "record"),
        messages.ImportAudioRequested("command", "audio"),
        messages.BarcodeCommitted("command", "scanner", "SN"),
        messages.ManualLabelRequested("command", "record", "OK"),
        messages.ManualAnalysisRequested("command", "record"),
        messages.BeginRecordingRequested("command", "session", False, {}),
        messages.RecordingMarkActionRequested("mark-command", 0),
        messages.LoadImportedAudioRequested("command", "import", "audio", None, {}),
        messages.AnalysisRequested("analysis", "source", {}, {}, False),
        messages.ExportRequested("job", "record", {}, ()),
        messages.PrepareAnalysisExportRequested(
            "request-analysis", "analysis", "source", "record", 0, {}, {}
        ),
        messages.PrepareManualLabelExportRequested(
            "request-label", "command", "record", "OK", 0
        ),
        messages.CancelExportPreparationRequested(
            "request-cancel", 0, "cancel"
        ),
        messages.CancelWorkflowRequested("command", 0, "cancel"),
        messages.CancelImportedAudioRequested("import", 0, "cancel"),
        messages.CancelRecordingRequested("session", 0, "cancel"),
        messages.CancelAnalysisRequested("analysis", 0, "cancel"),
        messages.CancelExportRequested("job", 0, "cancel"),
        messages.CommitRecordingLabelRequested("command", "record", "OK", ()),
        messages.RetryExportRequested("job", "attempt"),
        messages.IgnoreExportFailureRequested("job", "attempt"),
        messages.RetryShutdownFlushRequested(0, "job", "attempt"),
        messages.IgnoreShutdownFlushFailureRequested(0, "job", "attempt"),
        messages.ResourceLifecycleRequested(0, "disconnect-domains"),
        messages.AudioBatch(
            "session", 0, 0, 1, np.zeros((1, 1), dtype=np.float32), (0,)
        ),
        messages.AudioCompleted("session", -1, 0),
        messages.AudioFailed("session", -1, "failed", "failure"),
        messages.AudioCancelled("session", -1, "cancelled"),
        messages.RecordingStarted("session", {}),
        messages.RecordingBatchReady(
            "session", 0, 0, 1, np.zeros(1, dtype=np.float32)
        ),
        messages.RecordingCompleted("session", 1, {}),
        messages.RecordingFailed("session", "failure"),
        messages.RecordingCancelled("session", "cancelled"),
        messages.ImportedAudioReady("import", {}),
        messages.ImportedAudioFailed("import", "failure"),
        messages.AnalysisCompleted("analysis", "source", {}),
        messages.AnalysisExportPrepared(
            "request-analysis", "analysis", "source", "record", 0, {}, ()
        ),
        messages.AnalysisExportPreparationFailed(
            "request-analysis-failed",
            "analysis",
            "source",
            "record",
            0,
            "failure",
        ),
        messages.ManualLabelExportPrepared(
            "request-label", "command", "record", "OK", 0, {}, ()
        ),
        messages.ManualLabelExportPreparationFailed(
            "request-label-failed",
            "command",
            "record",
            "NG",
            0,
            "failure",
        ),
        messages.ExportPreparationCancelled("request-cancel", 0),
        messages.AnalysisTransportReady(
            "analysis", "source", "record", 0, {"Label": "OK"}
        ),
        messages.AnalysisFailed("analysis", "source", "failure"),
        messages.ExportCompleted("job", "attempt", "record", ()),
        messages.ExportFailed("job", "attempt", "record", ()),
        messages.ExportRetryAccepted(
            "job", "attempt-1", "attempt-2", 2
        ),
        messages.RecordingLabelCommitted("command", "record", "OK", ()),
        messages.RecordingLabelCommitFailed("command", "record", "OK", "failure"),
        messages.WorkflowCommandRejected("command", "IDLE", "busy"),
        messages.WorkflowStateChanged(
            0,
            "IDLE",
            "PREPARING",
            active_session_id="session",
            active_import_id="import",
            active_analysis_id="analysis",
            active_job_id="job",
        ),
        messages.ShutdownFlushFailed(0, "job", "attempt", ()),
    ]


def test_every_public_message_dataclass_uses_slot_backed_field_storage():
    instances = identifier_messages() + [
        messages.ConfigurationSnapshot({}, {}),
        messages.ShutdownRequested(0, False),
        messages.ConfirmShutdownCancellationRequested(0),
        messages.AbortShutdownRequested(0),
        messages.BeginShutdownFlushRequested(0),
        messages.ConfigurationChanged(0, {}),
        messages.ShutdownFlushCompleted(0),
        messages.ShutdownAborted(0),
        messages.ShutdownReady(0),
    ]
    declared = {
        message_type
        for message_type in vars(messages).values()
        if isinstance(message_type, type)
        and message_type.__module__ == messages.__name__
        and is_dataclass(message_type)
    }

    assert {type(instance) for instance in instances} == declared
    for instance in instances:
        assert type(instance).__bases__ == (messages._SealedMessage,)
        assert not hasattr(instance, "__dict__"), type(instance).__name__
        with pytest.raises(AttributeError):
            object.__setattr__(instance, "__dict__", {"replacement": True})

    batch = next(instance for instance in instances if type(instance) is AudioBatch)
    assert batch.multi.shape == (1, 1)
    assert batch.channel_order == (0,)


IDENTIFIER_CASES = [
    (message, field.name)
    for message in identifier_messages()
    for field in fields(message)
    if field.name.endswith("_id") and getattr(message, field.name) is not None
]


def test_identifier_validation_table_covers_every_message_identifier_field():
    declared = {
        (message_type.__name__, field.name)
        for message_type in vars(messages).values()
        if isinstance(message_type, type)
        and message_type.__module__ == messages.__name__
        and is_dataclass(message_type)
        for field in fields(message_type)
        if field.name.endswith("_id")
    }
    covered = {(type(message).__name__, field_name) for message, field_name in IDENTIFIER_CASES}

    assert covered == declared


@pytest.mark.parametrize(
    "message, field_name",
    IDENTIFIER_CASES,
    ids=lambda value: type(value).__name__ if is_dataclass(value) else value,
)
@pytest.mark.parametrize("invalid_identifier", ["", "   ", 7])
def test_every_present_identifier_requires_a_non_empty_string(
    message, field_name, invalid_identifier
):
    with pytest.raises(ValueError):
        replace(message, **{field_name: invalid_identifier})


@pytest.mark.parametrize("message, field_name", IDENTIFIER_CASES)
def test_every_present_identifier_rejects_string_subclasses(message, field_name):
    with pytest.raises(ValueError):
        replace(message, **{field_name: MutableText(getattr(message, field_name))})


@pytest.mark.parametrize(
    "message, field_name",
    [case for case in IDENTIFIER_CASES if not case[1].startswith("active_")],
)
def test_required_identifier_fields_reject_none(message, field_name):
    with pytest.raises(ValueError):
        replace(message, **{field_name: None})


SOURCE_CASES = [
    (messages.StartTestRequested("command", "manual", "SN", False, 0), "source"),
    (messages.ReplayRequested("command", "manual", "record"), "source"),
    (messages.BarcodeCommitted("command", "scanner", "SN"), "source"),
]


@pytest.mark.parametrize("message, field_name", SOURCE_CASES)
@pytest.mark.parametrize("invalid_source", ["", "   ", [], 1, None])
def test_required_source_tokens_are_non_empty_strings(message, field_name, invalid_source):
    with pytest.raises(ValueError):
        replace(message, **{field_name: invalid_source})


BOOLEAN_CASES = [
    (messages.StartTestRequested("command", "manual", "SN", False, 0), "skip_sn_regex_validation"),
    (messages.BeginRecordingRequested("command", "session", False, {}), "replay"),
    (messages.AnalysisRequested("analysis", "source", {}, {}, False), "automatic"),
    (messages.ShutdownRequested(0, False), "has_active_workflow"),
    (messages.RecordingFailed("session", "failure", audio_committed=False), "audio_committed"),
]


def test_boolean_validation_table_covers_every_boolean_message_field():
    declared = {
        (message_type.__name__, field.name)
        for message_type in vars(messages).values()
        if isinstance(message_type, type)
        and message_type.__module__ == messages.__name__
        and is_dataclass(message_type)
        for field in fields(message_type)
        if field.type in (bool, "bool")
    }
    covered = {(type(message).__name__, field_name) for message, field_name in BOOLEAN_CASES}

    assert covered == declared


@pytest.mark.parametrize("message, field_name", BOOLEAN_CASES)
@pytest.mark.parametrize("invalid_boolean", [0, 1.0, "false", None, [], np.int64(1)])
def test_boolean_control_fields_reject_truthy_and_falsy_non_booleans(
    message, field_name, invalid_boolean
):
    with pytest.raises(ValueError):
        replace(message, **{field_name: invalid_boolean})


@pytest.mark.parametrize("message, field_name", BOOLEAN_CASES)
@pytest.mark.parametrize("valid_boolean", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_control_fields_normalize_python_and_numpy_booleans(
    message, field_name, valid_boolean
):
    normalized = replace(message, **{field_name: valid_boolean})

    assert type(getattr(normalized, field_name)) is bool
    assert getattr(normalized, field_name) is bool(valid_boolean)


GENERATION_CASES = [
    (
        messages.RecordingMarkActionRequested("mark-command", 0),
        "workflow_generation",
    ),
    (
        messages.PrepareAnalysisExportRequested(
            "request", "analysis", "source", "record", 0, {}, {}
        ),
        "workflow_generation",
    ),
    (
        messages.PrepareManualLabelExportRequested(
            "request", "command", "record", "OK", 0
        ),
        "workflow_generation",
    ),
    (
        messages.CancelExportPreparationRequested("request", 0, "cancel"),
        "workflow_generation",
    ),
    (
        messages.AnalysisExportPrepared(
            "request", "analysis", "source", "record", 0, {}, ()
        ),
        "workflow_generation",
    ),
    (
        messages.AnalysisExportPreparationFailed(
            "request", "analysis", "source", "record", 0, "failed"
        ),
        "workflow_generation",
    ),
    (
        messages.ManualLabelExportPrepared(
            "request", "command", "record", "OK", 0, {}, ()
        ),
        "workflow_generation",
    ),
    (
        messages.ManualLabelExportPreparationFailed(
            "request", "command", "record", "NG", 0, "failed"
        ),
        "workflow_generation",
    ),
    (
        messages.ExportPreparationCancelled("request", 0),
        "workflow_generation",
    ),
    (messages.LoadImportedAudioRequested("command", "import", "audio", None, {}), "workflow_generation"),
    (messages.AnalysisRequested("analysis", "source", {}, {}, False), "workflow_generation"),
    (
        messages.AnalysisTransportReady(
            "analysis", "source", "record", 0, {"Label": "OK"}
        ),
        "workflow_generation",
    ),
    (messages.StartTestRequested("command", "manual", "SN", False, 0), "configuration_generation"),
    (messages.CancelWorkflowRequested("command", 0, "cancel"), "workflow_generation"),
    (messages.CancelImportedAudioRequested("import", 0, "cancel"), "workflow_generation"),
    (messages.CancelRecordingRequested("session", 0, "cancel"), "workflow_generation"),
    (messages.CancelAnalysisRequested("analysis", 0, "cancel"), "workflow_generation"),
    (messages.CancelExportRequested("job", 0, "cancel"), "workflow_generation"),
    (messages.ShutdownRequested(0, False), "shutdown_generation"),
    (messages.ConfirmShutdownCancellationRequested(0), "shutdown_generation"),
    (messages.AbortShutdownRequested(0), "shutdown_generation"),
    (messages.BeginShutdownFlushRequested(0), "shutdown_generation"),
    (messages.RetryShutdownFlushRequested(0, "job", "attempt"), "shutdown_generation"),
    (messages.IgnoreShutdownFlushFailureRequested(0, "job", "attempt"), "shutdown_generation"),
    (messages.WorkflowStateChanged(0, "IDLE", "CLOSING"), "workflow_generation"),
    (messages.ConfigurationChanged(0, {}), "configuration_generation"),
    (messages.ShutdownFlushFailed(0, "job", "attempt", ()), "shutdown_generation"),
    (messages.ShutdownFlushCompleted(0), "shutdown_generation"),
    (messages.ShutdownAborted(0), "shutdown_generation"),
    (messages.ShutdownReady(0), "shutdown_generation"),
    (
        messages.ResourceLifecycleRequested(0, "disconnect-domains"),
        "shutdown_generation",
    ),
]


def test_generation_validation_table_covers_every_generation_field():
    declared = {
        (message_type.__name__, field.name)
        for message_type in vars(messages).values()
        if isinstance(message_type, type)
        and message_type.__module__ == messages.__name__
        and is_dataclass(message_type)
        for field in fields(message_type)
        if field.name.endswith("_generation")
    }
    covered = {(type(message).__name__, field_name) for message, field_name in GENERATION_CASES}

    assert covered == declared


@pytest.mark.parametrize("message, field_name", GENERATION_CASES)
@pytest.mark.parametrize(
    "invalid_generation", [True, np.bool_(False), 1.0, np.nan, -1, "1", None]
)
def test_generation_fields_reject_non_integer_or_negative_values(
    message, field_name, invalid_generation
):
    with pytest.raises(ValueError):
        replace(message, **{field_name: invalid_generation})


@pytest.mark.parametrize("message, field_name", GENERATION_CASES)
def test_generation_fields_accept_numpy_integers_and_normalize_to_python_int(
    message, field_name
):
    normalized = replace(message, **{field_name: np.int64(4)})

    assert type(getattr(normalized, field_name)) is int
    assert getattr(normalized, field_name) == 4


def test_worker_thread_queued_bus_delivery_runs_on_receiver_thread_in_order(qapp):
    from PyQt5.QtCore import QEventLoop, QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

    class Receiver(QObject):
        complete = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.values = []
            self.thread_ids = []

        @pyqtSlot(object)
        def receive(self, value):
            self.values.append(value)
            self.thread_ids.append(int(QThread.currentThreadId()))
            if len(self.values) == 3:
                self.complete.emit()

    class WorkerEmitter(QObject):
        def __init__(self, signal):
            super().__init__()
            self.signal = signal

        @pyqtSlot()
        def run(self):
            for value in range(3):
                self.signal.emit(value)
            QThread.currentThread().quit()

    bus = SequenceEventBus()
    receiver = Receiver()
    worker_thread = QThread()
    emitter = WorkerEmitter(bus.events.workflow_state_changed)
    emitter.moveToThread(worker_thread)
    worker_thread.started.connect(emitter.run)
    worker_thread.finished.connect(emitter.deleteLater)
    bus.events.workflow_state_changed.connect(receiver.receive, Qt.QueuedConnection)
    event_loop = QEventLoop()
    timed_out = []
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: (timed_out.append(True), event_loop.quit()))
    receiver.complete.connect(event_loop.quit)
    main_thread_id = int(QThread.currentThreadId())

    worker_thread.start()
    timer.start(2000)
    event_loop.exec()
    timer.stop()
    worker_thread.quit()

    assert worker_thread.wait(2000)
    assert timed_out == []
    assert receiver.values == [0, 1, 2]
    assert receiver.thread_ids == [main_thread_id] * 3


def test_disconnected_and_destroyed_receivers_ignore_late_worker_signal(qapp):
    from PyQt5 import sip
    from PyQt5.QtCore import QObject, QThread, Qt, pyqtSlot

    class Receiver(QObject):
        def __init__(self, sink, name):
            super().__init__()
            self.sink = sink
            self.name = name

        @pyqtSlot(object)
        def receive(self, value):
            self.sink.append((self.name, value))

    class WorkerEmitter(QObject):
        def __init__(self, signal):
            super().__init__()
            self.signal = signal

        @pyqtSlot()
        def run(self):
            self.signal.emit("late")
            QThread.currentThread().quit()

    calls = []
    bus = SequenceEventBus()
    disconnected = Receiver(calls, "disconnected")
    destroyed = Receiver(calls, "destroyed")
    signal = bus.events.workflow_state_changed
    signal.connect(disconnected.receive, Qt.QueuedConnection)
    signal.disconnect(disconnected.receive)
    signal.connect(destroyed.receive, Qt.QueuedConnection)
    worker_thread = QThread()
    emitter = WorkerEmitter(signal)
    emitter.moveToThread(worker_thread)
    worker_thread.started.connect(emitter.run)
    worker_thread.finished.connect(emitter.deleteLater)

    worker_thread.start()
    assert worker_thread.wait(2000)
    sip.delete(destroyed)
    qapp.processEvents()

    assert calls == []


@pytest.mark.parametrize(
    "unsafe_dtype",
    [
        np.dtype("i4", metadata={"state": {"values": []}}),
        np.dtype(
            [("sample", np.dtype("i4", metadata={"state": {"values": []}}))]
        ),
        np.dtype([("payload", object)]),
        np.dtype([("sample", "f4")]),
        np.dtype("U4"),
        np.dtype("bool"),
    ],
)
def test_numpy_arrays_reject_non_numeric_or_schema_bearing_dtypes(unsafe_dtype):
    array = np.zeros(1, dtype=unsafe_dtype)

    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"unsafe": array})


@pytest.mark.parametrize(
    "unsafe_scalar",
    [
        np.array(
            (1,),
            dtype=np.dtype(
                [("sample", "i4")],
                metadata={"state": {"values": []}},
            ),
        )[()],
        np.array(
            (1,),
            dtype=np.dtype(
                [("sample", np.dtype("i4", metadata={"state": {"values": []}}))]
            ),
        )[()],
        np.array(
            ({"values": []},),
            dtype=np.dtype([("payload", object)]),
        )[()],
    ],
)
def test_numpy_scalars_reject_object_or_metadata_bearing_dtypes(unsafe_scalar):
    assert isinstance(unsafe_scalar, np.generic)

    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"unsafe": unsafe_scalar})


def test_structured_numpy_scalar_is_rejected_because_dtype_names_are_mutable():
    source = np.array([(3, 1.5)], dtype=[("sample", "i4"), ("level", "f4")])
    scalar = source[0]

    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"scalar": scalar})


def test_exact_simple_numpy_dtype_is_copied_as_a_supported_payload():
    source = np.dtype("i4")

    event = ConfigurationChanged(1, {"dtype": source})

    assert event.configuration_snapshot["dtype"] == source
    assert type(event.configuration_snapshot["dtype"]) is type(source)


def test_structured_numpy_dtype_is_rejected_from_payload_protocol():
    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"dtype": np.dtype([("sample", "i4")])})


def test_supported_numpy_numeric_scalars_normalize_to_safe_python_values():
    event = ConfigurationChanged(
        1,
        {
            "integer": np.int64(7),
            "floating": np.float32(1.5),
            "boolean": np.bool_(True),
        },
    )

    assert type(event.configuration_snapshot["integer"]) is int
    assert type(event.configuration_snapshot["floating"]) is float
    assert type(event.configuration_snapshot["boolean"]) is bool


def test_numpy_scalar_subclass_is_rejected_before_dtype_override():
    with pytest.raises(TypeError, match="exact supported NumPy scalar"):
        ConfigurationChanged(1, {"scalar": MaliciousFloat(1.5)})


@pytest.mark.parametrize(
    "enum_value",
    [SemanticKind.SESSION, MutableValueKind.SESSION, GlobalStatePropertyKind.SESSION],
    ids=["primitive-valued", "mutable-valued", "stateful-property"],
)
def test_all_enum_payloads_are_rejected_in_favor_of_primitive_tokens(enum_value):
    with pytest.raises(TypeError, match="primitive stable token"):
        ConfigurationChanged(1, {"enum": enum_value})


def test_enum_name_and_value_tokens_are_detached_plain_primitives():
    event = ConfigurationChanged(
        1,
        {"name": SemanticKind.SESSION.name, "value": SemanticKind.SESSION.value},
    )

    assert event.configuration_snapshot == {"name": "SESSION", "value": "session"}
    assert type(event.configuration_snapshot["name"]) is str
    assert type(event.configuration_snapshot["value"]) is str


@pytest.mark.parametrize("metadata_name", ["shape", "strides", "dtype", "data"])
def test_published_array_blocks_direct_metadata_assignment(metadata_name):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot["array"]
    replacement = {
        "shape": (6,),
        "strides": (4, 8),
        "dtype": np.dtype("i4"),
        "data": memoryview(bytes(source.nbytes)),
    }[metadata_name]

    with pytest.raises(AttributeError):
        setattr(frozen, metadata_name, replacement)

    assert frozen.shape == (2, 3)
    assert frozen.strides == (12, 4)
    assert frozen.dtype == np.dtype("float32")
    assert np.array_equal(frozen, source)
    assert frozen.flags.writeable is False


def test_published_array_blocks_in_place_resize_metadata_mutation():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot["array"]

    with pytest.raises(AttributeError):
        frozen.resize((6,), refcheck=False)

    assert frozen.shape == (2, 3)
    assert frozen.strides == (12, 4)
    assert frozen.dtype == np.dtype("float32")
    assert np.array_equal(frozen, source)


def test_published_array_blocks_pickle_state_replacement():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot["array"]
    original_base = frozen.base
    hostile_state = (
        1,
        (3,),
        np.dtype("int64"),
        False,
        np.full(3, 9, dtype=np.int64).tobytes(),
    )

    with pytest.raises(TypeError, match="deserialization"):
        frozen.__setstate__(hostile_state)

    assert frozen.shape == (2, 3)
    assert frozen.strides == (12, 4)
    assert frozen.dtype == np.dtype("float32")
    assert frozen.base is original_base
    assert frozen.flags.writeable is False
    assert np.array_equal(frozen, source)


def test_published_array_pickle_policy_is_stable_rejection():
    frozen = ConfigurationChanged(
        1, {"array": np.arange(3, dtype=np.float32)}
    ).configuration_snapshot["array"]

    with pytest.raises(TypeError, match="pickling"):
        pickle.dumps(frozen)


def test_published_array_flags_are_read_only_but_remain_inspectable():
    frozen = ConfigurationChanged(
        1, {"array": np.arange(6, dtype=np.float32).reshape(2, 3)}
    ).configuration_snapshot["array"]

    assert frozen.flags.writeable is False
    assert frozen.flags.c_contiguous is True
    assert frozen.flags.f_contiguous is False
    assert frozen.flags.aligned is True
    assert frozen.flags.owndata is False
    assert frozen.flags["C_CONTIGUOUS"] is True
    assert frozen.flags["WRITEABLE"] is False
    assert "C_CONTIGUOUS" in repr(frozen.flags)

    with pytest.raises(AttributeError):
        frozen.flags.aligned = False
    with pytest.raises(AttributeError):
        frozen.flags.writeable = True
    with pytest.raises(TypeError):
        frozen.flags["ALIGNED"] = False
    with pytest.raises(ValueError):
        frozen.setflags(align=False)


def test_published_array_flags_match_numpy_read_only_inspection_parity():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = ConfigurationChanged(1, {"array": source}).configuration_snapshot["array"]
    standard = np.ndarray(
        shape=source.shape,
        dtype=source.dtype,
        buffer=bytes(source.tobytes()),
    )
    properties = (
        "c_contiguous",
        "contiguous",
        "f_contiguous",
        "fortran",
        "owndata",
        "writeable",
        "aligned",
        "writebackifcopy",
        "fnc",
        "forc",
        "behaved",
        "carray",
        "farray",
        "num",
    )
    keys = (
        "C",
        "CONTIGUOUS",
        "C_CONTIGUOUS",
        "F",
        "FORTRAN",
        "F_CONTIGUOUS",
        "O",
        "OWNDATA",
        "W",
        "WRITEABLE",
        "A",
        "ALIGNED",
        "X",
        "WRITEBACKIFCOPY",
        "FNC",
        "FORC",
        "B",
        "BEHAVED",
        "CA",
        "CARRAY",
        "FA",
        "FARRAY",
    )

    assert {
        name: getattr(frozen.flags, name) for name in properties
    } == {name: getattr(standard.flags, name) for name in properties}
    assert {key: frozen.flags[key] for key in keys} == {
        key: standard.flags[key] for key in keys
    }


@pytest.mark.parametrize(
    "view_factory",
    [np.asarray, lambda value: value.view(np.ndarray)],
    ids=["asarray", "base-view"],
)
def test_base_array_views_cannot_change_published_array_invariants(view_factory):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    frozen = ConfigurationChanged(
        1, {"array": expected}
    ).configuration_snapshot["array"]
    base_view = view_factory(frozen)

    base_view.shape = (6,)
    base_view.dtype = np.dtype("i4")
    with pytest.raises(ValueError):
        base_view[:] = 0

    assert frozen.shape == (2, 3)
    assert frozen.dtype == np.dtype("float32")
    assert np.array_equal(frozen, expected)
    assert frozen.flags.writeable is False


def test_payload_freezing_rejects_self_referential_list():
    cycle = []
    cycle.append(cycle)

    with pytest.raises((TypeError, ValueError), match="cyclic"):
        ConfigurationChanged(1, {"payload": cycle})


def test_payload_freezing_rejects_self_referential_dict():
    cycle = {}
    cycle["self"] = cycle

    with pytest.raises((TypeError, ValueError), match="cyclic"):
        ConfigurationChanged(1, {"payload": cycle})


def test_payload_freezing_rejects_mutual_container_cycle():
    left = []
    right = {"left": left}
    left.append(right)

    with pytest.raises((TypeError, ValueError), match="cyclic"):
        ConfigurationChanged(1, {"payload": left})


def test_configuration_snapshot_constructor_rejects_cyclic_payload():
    cycle = {}
    cycle["self"] = cycle

    with pytest.raises((TypeError, ValueError), match="cyclic"):
        ConfigurationSnapshot(sequence_config=cycle, analysis_config={})


def test_payload_freezing_allows_repeated_non_cyclic_references():
    shared = [{"values": [1, 2]}]

    event = ConfigurationChanged(1, {"first": shared, "second": shared})
    first = event.configuration_snapshot["first"]
    second = event.configuration_snapshot["second"]

    assert first == second
    assert first is not second
    assert first[0] is not second[0]
    assert first[0]["values"] is not second[0]["values"]


@pytest.mark.parametrize(
    "payload, calls",
    [
        (HookedDict(value=1), HookedDict.calls),
        (HookedList((1, 2)), HookedList.calls),
        (HookedTuple((1, 2)), HookedTuple.calls),
        (HookedSet((1, 2)), HookedSet.calls),
        (HookedMapping(), HookedMapping.calls),
    ],
    ids=["dict-subclass", "list-subclass", "tuple-subclass", "set-subclass", "mapping"],
)
def test_payload_container_subclasses_are_rejected_without_hooks(payload, calls):
    calls.clear()

    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"payload": payload})

    assert calls == []


def test_payload_protocol_accepts_only_audited_exact_container_representations():
    event = ConfigurationChanged(
        1,
        {
            "dict": {"value": 1},
            "list": [1, 2],
            "tuple": (1, 2),
            "set": {1, 2},
            "frozenset": frozenset((1, 2)),
            "posix": PurePosixPath("recordings/session.wav"),
            "windows": PureWindowsPath("recordings/session.wav"),
        },
    )

    frozen = event.configuration_snapshot
    assert type(frozen) is messages._FrozenMapping
    assert type(frozen["dict"]) is messages._FrozenMapping
    assert frozen["dict"] == {"value": 1}
    assert dict(frozen["dict"]) == {"value": 1}
    assert tuple(frozen["dict"]) == ("value",)
    assert type(frozen["list"]) is tuple
    assert type(frozen["tuple"]) is tuple
    assert type(frozen["set"]) is frozenset
    assert type(frozen["frozenset"]) is frozenset
    assert type(frozen["posix"]) is PurePosixPath
    assert type(frozen["windows"]) is PureWindowsPath


def test_frozen_mapping_blocks_assignment_and_deletion_without_data_loss():
    frozen = ConfigurationChanged(1, {"value": 1}).configuration_snapshot

    with pytest.raises(AttributeError, match="immutable"):
        frozen._items = ()
    with pytest.raises(AttributeError, match="immutable"):
        del frozen._items
    with pytest.raises(TypeError):
        frozen["value"] = 2

    assert frozen["value"] == 1
    assert dict(frozen) == {"value": 1}


def test_external_frozen_mapping_is_recursively_refrozen_and_alias_isolated():
    mutable_list = [1, 2]
    mutable_dict = {"nested": [3, 4]}
    external = messages._FrozenMapping(
        (("list", mutable_list), ("dict", mutable_dict))
    )

    admitted = ConfigurationChanged(1, {"payload": external}).configuration_snapshot[
        "payload"
    ]
    mutable_list.append(5)
    mutable_dict["nested"].append(6)

    assert admitted is not external
    assert admitted["list"] == (1, 2)
    assert admitted["dict"]["nested"] == (3, 4)


def test_external_frozen_mapping_rejects_unsupported_nested_value():
    external = messages._FrozenMapping((("unsafe", object()),))

    with pytest.raises(TypeError, match="unsupported mutable message payload"):
        ConfigurationChanged(1, {"payload": external})


def test_forged_frozen_mapping_self_cycle_is_rejected():
    cycle = object.__new__(messages._FrozenMapping)
    items = (("self", cycle),)
    object.__setattr__(cycle, "_items", items)
    object.__setattr__(cycle, "_lookup", MappingProxyType(dict(items)))

    with pytest.raises(TypeError, match="cyclic"):
        ConfigurationChanged(1, {"payload": cycle})


def test_forged_frozen_mapping_mutual_cycle_is_rejected():
    left = object.__new__(messages._FrozenMapping)
    right = object.__new__(messages._FrozenMapping)
    left_items = (("right", right),)
    right_items = (("left", left),)
    object.__setattr__(left, "_items", left_items)
    object.__setattr__(left, "_lookup", MappingProxyType(dict(left_items)))
    object.__setattr__(right, "_items", right_items)
    object.__setattr__(right, "_lookup", MappingProxyType(dict(right_items)))

    with pytest.raises(TypeError, match="cyclic"):
        ConfigurationChanged(1, {"payload": left})


def test_queued_frozen_mapping_cannot_be_deleted_before_delivery(qapp):
    from PyQt5.QtCore import Qt

    event = ConfigurationChanged(1, {"nested": {"value": 1}})
    nested = event.configuration_snapshot["nested"]
    bus = SequenceEventBus()
    received = []
    bus.events.configuration_changed.connect(received.append, Qt.QueuedConnection)

    bus.events.configuration_changed.emit(event)
    with pytest.raises(AttributeError, match="immutable"):
        del nested._items
    with pytest.raises(AttributeError, match="immutable"):
        del nested._lookup
    qapp.processEvents()

    assert received[0].configuration_snapshot["nested"]["value"] == 1


def test_frozen_mapping_uses_constant_lookup_and_linear_mapping_conversion():
    keys = [CountingLookupKey(index) for index in range(64)]
    frozen = messages._FrozenMapping(
        tuple((key, index) for index, key in enumerate(keys))
    )
    lookup = object.__getattribute__(frozen, "_lookup")
    for key in keys:
        key.hash_calls = 0
        key.equality_calls = 0

    assert type(lookup) is MappingProxyType
    assert frozen[keys[-1]] == 63
    assert sum(key.hash_calls for key in keys) <= 2
    assert sum(key.equality_calls for key in keys) <= 1

    for key in keys:
        key.hash_calls = 0
        key.equality_calls = 0
    assert dict(frozen) == {key: index for index, key in enumerate(keys)}
    assert sum(key.hash_calls for key in keys) <= len(keys) * 4


def test_external_mapping_proxy_is_rejected_without_invoking_backing_mapping():
    backing = HookedMapping()
    proxy = MappingProxyType(backing)
    HookedMapping.calls.clear()

    with pytest.raises(TypeError, match="mapping proxy"):
        ConfigurationChanged(1, {"proxy": proxy})

    assert HookedMapping.calls == []


def test_payload_protocol_rebuilds_exact_decimal_and_temporal_values():
    fixed_zone = timezone(timedelta(hours=8), "China Standard Time")
    values = {
        "decimal": Decimal("1.25"),
        "date": date(2026, 8, 19),
        "naive_datetime": datetime(2026, 8, 19, 10, 30, fold=1),
        "fixed_datetime": datetime(
            2026, 8, 19, 10, 30, tzinfo=fixed_zone, fold=1
        ),
        "naive_time": time(10, 30, fold=1),
        "fixed_time": time(10, 30, tzinfo=fixed_zone, fold=1),
        "timedelta": timedelta(seconds=3),
        "timezone": fixed_zone,
    }

    event = ConfigurationChanged(1, values)

    assert dict(event.configuration_snapshot) == values
    for name, value in values.items():
        assert type(event.configuration_snapshot[name]) is type(value)
        if type(value) in (date, datetime, time, timedelta, timezone):
            assert event.configuration_snapshot[name] is not value

    assert event.configuration_snapshot["naive_datetime"].fold == 1
    assert event.configuration_snapshot["fixed_datetime"].fold == 1
    assert event.configuration_snapshot["naive_time"].fold == 1
    assert event.configuration_snapshot["fixed_time"].fold == 1


def test_timezone_rebuild_detaches_offset_and_name_subclasses():
    offset = MutableTimezoneOffset(hours=8)
    offset.values = []
    name = MutableTimezoneName("China Standard Time")
    name.values = []
    source = timezone(offset, name)
    MutableTimezoneOffset.calls.clear()
    MutableTimezoneName.calls.clear()

    event = ConfigurationChanged(1, {"timezone": source})
    offset.values.append("changed")
    name.values.append("changed")
    frozen = event.configuration_snapshot["timezone"]
    frozen_offset = timezone.utcoffset(frozen, None)
    frozen_name = timezone.tzname(frozen, None)

    assert type(frozen) is timezone
    assert frozen is not source
    assert type(frozen_offset) is timedelta
    assert frozen_offset == timedelta(hours=8)
    assert type(frozen_name) is str
    assert frozen_name == "China Standard Time"
    assert not hasattr(frozen_offset, "values")
    assert not hasattr(frozen_name, "values")
    assert MutableTimezoneOffset.calls == []
    assert MutableTimezoneName.calls == []


@pytest.mark.parametrize("temporal_type", [datetime, time])
def test_custom_tzinfo_is_rejected_without_invoking_hooks(temporal_type):
    hostile_zone = HostileTimezone()
    if temporal_type is datetime:
        value = datetime(2026, 8, 19, 10, 30, tzinfo=hostile_zone)
    else:
        value = time(10, 30, tzinfo=hostile_zone)

    with pytest.raises(TypeError, match="exact datetime.timezone"):
        ConfigurationChanged(1, {"temporal": value})

    assert hostile_zone.calls == []


def test_queued_temporal_payload_is_detached_before_delivery(qapp):
    from PyQt5.QtCore import Qt

    offset = MutableTimezoneOffset(hours=8)
    offset.values = []
    name = MutableTimezoneName("China Standard Time")
    name.values = []
    source_timezone = timezone(offset, name)
    MutableTimezoneOffset.calls.clear()
    MutableTimezoneName.calls.clear()
    event = ConfigurationChanged(
        1,
        {
            "when": datetime(
                2026, 8, 19, 10, 30, tzinfo=source_timezone, fold=1
            )
        },
    )
    bus = SequenceEventBus()
    received = []
    bus.events.configuration_changed.connect(received.append, Qt.QueuedConnection)

    bus.events.configuration_changed.emit(event)
    offset.values.append("changed")
    name.values.append("changed")
    qapp.processEvents()

    delivered = received[0].configuration_snapshot["when"]
    delivered_zone = datetime.tzinfo.__get__(delivered)
    assert type(delivered) is datetime
    assert delivered.fold == 1
    assert type(delivered_zone) is timezone
    assert type(timezone.utcoffset(delivered_zone, None)) is timedelta
    assert type(timezone.tzname(delivered_zone, None)) is str
    assert MutableTimezoneOffset.calls == []
    assert MutableTimezoneName.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        SafeMethodAndConstantPayload(),
        DescriptorStatePayload(),
        PropertyGlobalStatePayload(),
    ],
    ids=[
        "safe-method-and-constant",
        "mutable-descriptor",
        "property-global",
    ],
)
def test_arbitrary_dataclass_behavior_is_not_inspected_or_preserved(payload):
    with pytest.raises(TypeError, match="explicit snapshot"):
        ConfigurationChanged(1, {"payload": payload})


@pytest.mark.parametrize(
    "payload, calls",
    [
        (HookedInt(4), HookedInt.calls),
        (HookedText("text"), HookedText.calls),
        (HookedPath("recordings/session.wav"), HookedPath.calls),
    ],
    ids=["int-subclass", "str-subclass", "path-subclass"],
)
def test_primitive_and_path_subclasses_are_rejected_without_hooks(payload, calls):
    calls.clear()

    with pytest.raises(TypeError):
        ConfigurationChanged(1, {"payload": payload})

    assert calls == []


TEXT_CASES = [
    (message, field.name)
    for message in identifier_messages()
    for field in fields(message)
    if field.type in (str, "str")
]


def test_text_validation_table_covers_every_direct_string_field():
    declared = {
        (message_type.__name__, field.name)
        for message_type in vars(messages).values()
        if isinstance(message_type, type)
        and message_type.__module__ == messages.__name__
        and is_dataclass(message_type)
        for field in fields(message_type)
        if field.type in (str, "str")
    }
    covered = {(type(message).__name__, field_name) for message, field_name in TEXT_CASES}

    assert covered == declared


@pytest.mark.parametrize("message, field_name", TEXT_CASES)
@pytest.mark.parametrize(
    "invalid_text",
    [[], ("text",), {"text": True}, MutableCustomPayload([]), 1, None],
)
def test_every_direct_string_field_rejects_non_string_values(
    message, field_name, invalid_text
):
    with pytest.raises(ValueError):
        replace(message, **{field_name: invalid_text})


@pytest.mark.parametrize("message, field_name", TEXT_CASES)
def test_every_direct_string_field_rejects_mutable_subclasses(message, field_name):
    with pytest.raises(ValueError):
        replace(message, **{field_name: MutableText(getattr(message, field_name))})


FREE_FORM_TEXT_CASES = [
    (message, field_name)
    for message, field_name in TEXT_CASES
    if not field_name.endswith("_id")
    and field_name not in {"source", "operation"}
]


@pytest.mark.parametrize("message, field_name", FREE_FORM_TEXT_CASES)
def test_free_form_text_fields_preserve_existing_empty_string_acceptance(
    message, field_name
):
    normalized = replace(message, **{field_name: ""})

    assert getattr(normalized, field_name) == ""


@pytest.mark.parametrize("wrong_multi", [[], (), 1, np.float32(1)])
def test_audio_batch_requires_multi_to_be_an_ndarray(wrong_multi):
    with pytest.raises(TypeError):
        AudioBatch("session", 0, 0, 1, wrong_multi, (0,))


def test_audio_batch_callback_factory_rejects_non_ndarray_multi():
    with pytest.raises(TypeError):
        AudioBatch.from_callback(
            session_id="session",
            sequence_no=0,
            sample_start=0,
            multi=[[1.0]],
            channel_order=(0,),
        )


def test_canonical_recording_admission_registry_is_exact_and_atomic():
    from PyQt5 import sip
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    workflow_owner = QObject()
    recording_owner = QObject()
    workflow_capability = bus._bind_canonical_recording_workflow_owner(
        workflow_owner
    )
    recording_capability = bus._bind_canonical_recording_consumer(
        recording_owner
    )
    canonical = BeginRecordingRequested(
        "canonical-command",
        "canonical-session",
        False,
        {"workflow_generation": 9},
    )
    collision = BeginRecordingRequested(
        "collision-command",
        "canonical-session",
        False,
        {"workflow_generation": 9},
    )

    assert not hasattr(bus, "register_canonical_recording_admission")
    assert not hasattr(bus, "claim_canonical_recording_terminal")
    assert not hasattr(bus, "retire_canonical_recording_admission")
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="capability"):
            copier(workflow_capability)
    assert (
        bus._register_canonical_recording_admission(object(), canonical)
        is False
    )
    other_bus = SequenceEventBus()
    assert (
        other_bus._register_canonical_recording_admission(
            workflow_capability,
            canonical,
        )
        is False
    )
    assert (
        bus._register_canonical_recording_admission(
            workflow_capability,
            canonical,
        )
        is True
    )
    assert (
        bus._register_canonical_recording_admission(
            workflow_capability,
            canonical,
        )
        is True
    )
    assert (
        bus._register_canonical_recording_admission(
            workflow_capability,
            collision,
        )
        is False
    )
    assert (
        bus._claim_canonical_recording_terminal(
            recording_capability,
            collision,
        )
        is None
    )
    begin_claim = bus._claim_canonical_recording_begin(
        recording_capability,
        canonical,
    )
    assert begin_claim is not None
    assert begin_claim.identity == ("canonical-session", 9)
    assert begin_claim.cancellation is None

    barrier = Barrier(9)
    claims = []

    def claim():
        barrier.wait()
        claims.append(
            bus._claim_canonical_recording_terminal(
                recording_capability,
                canonical,
            )
        )

    workers = [Thread(target=claim) for _index in range(8)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)

    assert claims.count(("canonical-session", 9)) == 1
    assert claims.count(None) == 7
    assert all(not worker.is_alive() for worker in workers)

    preserved = BeginRecordingRequested(
        "preserved-command",
        "preserved-session",
        False,
        {"workflow_generation": 10},
    )
    assert bus._register_canonical_recording_admission(
        workflow_capability,
        preserved,
    )
    sip.delete(recording_owner)
    replacement_recording_owner = QObject()
    replacement_recording_capability = (
        bus._bind_canonical_recording_consumer(replacement_recording_owner)
    )
    assert replacement_recording_capability is not None
    assert (
        bus._replay_canonical_recording_admission(
            replacement_recording_capability
        )
        is preserved
    )
    assert bus._claim_canonical_recording_begin(
        replacement_recording_capability,
        preserved,
    ) is not None
    assert bus._claim_canonical_recording_terminal(
        replacement_recording_capability,
        preserved,
    ) == ("preserved-session", 10)

    gc_preserved = BeginRecordingRequested(
        "gc-preserved-command",
        "gc-preserved-session",
        False,
        {"workflow_generation": 11},
    )
    assert bus._register_canonical_recording_admission(
        workflow_capability,
        gc_preserved,
    )
    replacement_recording_owner_ref = ref(replacement_recording_owner)
    del replacement_recording_owner
    gc.collect()
    assert replacement_recording_owner_ref() is None
    gc_replacement_owner = QObject()
    gc_replacement_capability = bus._bind_canonical_recording_consumer(
        gc_replacement_owner
    )
    assert bus._replay_canonical_recording_admission(
        gc_replacement_capability
    ) is gc_preserved
    assert bus._claim_canonical_recording_begin(
        gc_replacement_capability,
        gc_preserved,
    ) is not None
    assert bus._claim_canonical_recording_terminal(
        gc_replacement_capability,
        gc_preserved,
    ) == ("gc-preserved-session", 11)

    orphaned = BeginRecordingRequested(
        "orphaned-command",
        "orphaned-session",
        False,
        {"workflow_generation": 12},
    )
    assert bus._register_canonical_recording_admission(
        workflow_capability,
        orphaned,
    )
    sip.delete(workflow_owner)
    replacement_owner = QObject()
    replacement_capability = bus._bind_canonical_recording_workflow_owner(
        replacement_owner
    )
    replacement = BeginRecordingRequested(
        "replacement-command",
        "replacement-session",
        False,
        {"workflow_generation": 13},
    )
    assert replacement_capability is not None
    assert bus._register_canonical_recording_admission(
        replacement_capability,
        replacement,
    )


def test_standalone_recording_admission_requires_explicit_revocable_mode():
    from PyQt5.QtCore import QObject

    command = BeginRecordingRequested(
        "standalone-command",
        "standalone-session",
        False,
        {"workflow_generation": 7},
    )
    default_bus = SequenceEventBus()
    default_owner = QObject()
    default_capability = default_bus._bind_canonical_recording_consumer(
        default_owner
    )

    assert default_bus._claim_canonical_recording_begin(
        default_capability,
        command,
    ) is None

    standalone_bus = SequenceEventBus(standalone_recording_admission=True)
    standalone_owner = QObject()
    standalone_capability = (
        standalone_bus._bind_canonical_recording_consumer(standalone_owner)
    )
    claim = standalone_bus._claim_canonical_recording_begin(
        standalone_capability,
        command,
    )

    assert claim is not None
    assert claim.identity == ("standalone-session", 7)
    assert claim.workflow_admitted is False
    assert standalone_bus._release_canonical_recording_consumer(
        standalone_capability
    )
    replacement_owner = QObject()
    replacement_capability = (
        standalone_bus._bind_canonical_recording_consumer(replacement_owner)
    )
    assert standalone_bus._claim_canonical_recording_begin(
        replacement_capability,
        command,
    ) is None


def test_workflow_release_tombstone_rejects_old_exact_copy_and_generation_collision():
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus(standalone_recording_admission=True)
    registry = bus._canonical_recording_admission_registry
    registry._workflow_epoch = 2**63 - 1
    first_workflow_owner = QObject()
    recording_owner = QObject()
    first_workflow_capability = (
        bus._bind_canonical_recording_workflow_owner(first_workflow_owner)
    )
    assert registry._workflow_epoch == 2**63
    recording_capability = bus._bind_canonical_recording_consumer(
        recording_owner
    )
    old = BeginRecordingRequested(
        "shared-command",
        "shared-session",
        False,
        {"workflow_generation": 2**63 - 1},
    )
    old_copy = BeginRecordingRequested(
        old.command_id,
        old.session_id,
        old.replay,
        old.session_snapshot,
    )
    collision = BeginRecordingRequested(
        "colliding-command",
        old.session_id,
        old.replay,
        old.session_snapshot,
    )
    assert bus._register_canonical_recording_admission(
        first_workflow_capability,
        old,
    )
    assert bus._release_canonical_recording_workflow_owner(
        first_workflow_capability
    )

    for stale in (old, old_copy, collision):
        assert bus._claim_canonical_recording_begin(
            recording_capability,
            stale,
        ) is None

    replacement_workflow_owner = QObject()
    replacement_workflow_capability = (
        bus._bind_canonical_recording_workflow_owner(
            replacement_workflow_owner
        )
    )
    assert registry._workflow_epoch == 2**63 + 1
    replacement = BeginRecordingRequested(
        old.command_id,
        old.session_id,
        old.replay,
        old.session_snapshot,
    )
    assert bus._register_canonical_recording_admission(
        replacement_workflow_capability,
        replacement,
    )
    assert bus._claim_canonical_recording_begin(
        recording_capability,
        old,
    ) is None
    assert bus._claim_canonical_recording_begin(
        recording_capability,
        replacement,
    ) is not None


def test_workflow_owner_gc_retains_recording_admission_revocation():
    from PyQt5.QtCore import QObject

    bus = SequenceEventBus()
    workflow_owner = QObject()
    workflow_owner_ref = ref(workflow_owner)
    recording_owner = QObject()
    workflow_capability = bus._bind_canonical_recording_workflow_owner(
        workflow_owner
    )
    recording_capability = bus._bind_canonical_recording_consumer(
        recording_owner
    )
    command = BeginRecordingRequested(
        "gc-command",
        "gc-session",
        False,
        {"workflow_generation": 4},
    )
    assert bus._register_canonical_recording_admission(
        workflow_capability,
        command,
    )

    del workflow_owner
    gc.collect()

    assert workflow_owner_ref() is None
    assert bus._claim_canonical_recording_begin(
        recording_capability,
        command,
    ) is None


def test_concurrent_workflow_release_and_begin_claim_leave_no_post_release_claim():
    from PyQt5.QtCore import QObject

    for index in range(32):
        bus = SequenceEventBus()
        workflow_owner = QObject()
        recording_owner = QObject()
        workflow_capability = bus._bind_canonical_recording_workflow_owner(
            workflow_owner
        )
        recording_capability = bus._bind_canonical_recording_consumer(
            recording_owner
        )
        command = BeginRecordingRequested(
            f"race-command-{index}",
            f"race-session-{index}",
            False,
            {"workflow_generation": 2**63 + index},
        )
        assert bus._register_canonical_recording_admission(
            workflow_capability,
            command,
        )
        barrier = Barrier(3)
        releases = []
        claims = []

        def release_owner():
            barrier.wait()
            releases.append(
                bus._release_canonical_recording_workflow_owner(
                    workflow_capability
                )
            )

        def claim_begin():
            barrier.wait()
            claims.append(
                bus._claim_canonical_recording_begin(
                    recording_capability,
                    command,
                )
            )

        release_thread = Thread(target=release_owner)
        claim_thread = Thread(target=claim_begin)
        release_thread.start()
        claim_thread.start()
        barrier.wait()
        release_thread.join(2)
        claim_thread.join(2)

        assert not release_thread.is_alive()
        assert not claim_thread.is_alive()
        assert releases == [True]
        assert len(claims) == 1
        assert bus._claim_canonical_recording_begin(
            recording_capability,
            command,
        ) is None
        assert bus._release_canonical_recording_workflow_owner(
            workflow_capability
        ) is False


@pytest.mark.parametrize("wrong_mono", [[], (), 1, np.float32(1)])
def test_audio_batch_requires_optional_mono_to_be_an_ndarray(wrong_mono):
    with pytest.raises(TypeError):
        AudioBatch(
            "session",
            0,
            0,
            1,
            np.zeros((1, 1), dtype=np.float32),
            (0,),
            wrong_mono,
        )


@pytest.mark.parametrize("wrong_display", [[], (), 1, np.float32(1)])
def test_recording_batch_ready_requires_display_to_be_an_ndarray(wrong_display):
    with pytest.raises(TypeError):
        RecordingBatchReady("session", 0, 0, 1, wrong_display)
