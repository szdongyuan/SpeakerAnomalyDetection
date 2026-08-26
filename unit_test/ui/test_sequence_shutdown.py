import gc
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Event, RLock, Timer
from types import SimpleNamespace
import weakref

import pytest
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QShowEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from ui.sequence.sequence_event_bus import (
    SequenceEventBus,
    WorkflowContinuationDeliveryOutcome,
    WorkflowContinuationDeliveryStatus,
)
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import SequenceExportModel
from ui.sequence.sequence_export_model import SpoolTarget
from ui.sequence.sequence_export_view import SequenceExportView
from ui.sequence.sequence_messages import (
    BeginShutdownFlushRequested,
    ConfirmShutdownCancellationRequested,
    IgnoreShutdownFlushFailureRequested,
    ShutdownAborted,
    ShutdownFlushCompleted,
    ShutdownReady,
    ShutdownRequested,
    StartTestRequested,
    RecordingFailed,
    ResourceLifecycleRequested,
)
from ui.sequence.sequence_export_worker import SequenceExportWorker
from ui.sequence.sequence_workflow_controller import (
    SequenceShutdownCoordinator,
    SequenceWorkflowController,
)
from ui.sequence.sequence_workflow_model import (
    SequenceWorkflowModel,
    SessionOrigin,
    WorkflowPhase,
)
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_resource_lifecycle_controller import (
    SequenceResourceLifecycleController,
    SequenceResourceLifecycleModel,
    SequenceResourceLifecycleView,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _attach_resource_lifecycle_owner(widget):
    if not hasattr(widget, "configuration_model"):
        widget.configuration_model = SequenceConfigurationModel()
    model = SequenceResourceLifecycleModel()
    view = SequenceResourceLifecycleView(widget)
    controller = SequenceResourceLifecycleController(
        view,
        model,
        lifecycle_bus=getattr(widget, "sequence_event_bus", None),
        parent=widget,
    )
    widget.resource_lifecycle_model = model
    widget.resource_lifecycle_view = view
    widget.resource_lifecycle_controller = controller
    return controller


def test_lifecycle_model_resolves_only_provably_retired_production_owner(qapp):
    from PyQt5 import sip

    class Recipient(QObject):
        def __init__(self, result):
            super().__init__()
            self.result = result
            self.calls = []

        def disconnect(self, request):
            self.calls.append(request.shutdown_generation)
            return self.result

    bus = SequenceEventBus()
    model = SequenceResourceLifecycleModel()
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(), model, lifecycle_bus=bus
    )
    recipient = Recipient(False)
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "analysis",
        recipient.disconnect,
        owner=recipient,
    )
    other = Recipient(False)
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains", "analysis", token, other
    ) is False
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains", "analysis", token, recipient
    ) is True

    sip.delete(recipient)

    assert lifecycle._publish_lifecycle_request(31, "disconnect-domains") is True
    assert recipient.calls == []
    assert bus.pending_resource_lifecycle_request_count == 0
    assert bus.completed_resource_lifecycle_request_count == 1


def test_lifecycle_model_keeps_failed_owner_until_destroy_then_runs_replacement(qapp):
    from PyQt5 import sip

    class Recipient(QObject):
        def __init__(self, name, result):
            super().__init__()
            self.name = name
            self.result = result
            self.calls = []

        def disconnect(self, request):
            self.calls.append(request.shutdown_generation)
            return self.result

    bus = SequenceEventBus()
    model = SequenceResourceLifecycleModel()
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(), model, lifecycle_bus=bus
    )
    old = Recipient("old", False)
    old_token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains", "workflow", old.disconnect, owner=old
    )
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains", "workflow", old_token, old
    )
    assert lifecycle._publish_lifecycle_request(32, "disconnect-domains") is False
    assert old.calls == [32]

    sip.delete(old)
    replacement = Recipient("replacement", True)
    replacement_token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        replacement.disconnect,
        owner=replacement,
    )
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains",
        "workflow",
        replacement_token,
        replacement,
    )

    assert lifecycle._publish_lifecycle_request(32, "disconnect-domains") is True
    assert replacement.calls == []
    assert lifecycle._publish_lifecycle_request(33, "disconnect-domains") is True
    assert replacement.calls == [33]
    assert bus.pending_resource_lifecycle_request_count == 0


def test_production_lifecycle_retry_reuses_original_request_identity(qapp):
    class Recipient(QObject):
        def __init__(self):
            super().__init__()
            self.ready = False
            self.requests = []

        def disconnect(self, request):
            self.requests.append(request)
            return self.ready

    bus = SequenceEventBus()
    model = SequenceResourceLifecycleModel()
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(), model, lifecycle_bus=bus
    )
    recipient = Recipient()
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        recipient.disconnect,
        owner=recipient,
    )
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains", "workflow", token, recipient
    )

    assert lifecycle._publish_lifecycle_request(37, "disconnect-domains") is False
    recipient.ready = True
    assert lifecycle._publish_lifecycle_request(37, "disconnect-domains") is True

    assert len(recipient.requests) == 2
    assert recipient.requests[0] is recipient.requests[1]


def test_production_registration_handles_native_delete_and_exact_replacement(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    class Recipient(QObject):
        def __init__(self, name, result=True):
            super().__init__()
            self.name = name
            self.result = result
            self.calls = []

        def disconnect(self, request):
            self.calls.append(request.shutdown_generation)
            return self.result

    sequence = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(sequence)
    sequence.sequence_event_bus = SequenceEventBus(sequence)
    sequence.resource_lifecycle_model = SequenceResourceLifecycleModel()
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(),
        sequence.resource_lifecycle_model,
        lifecycle_bus=sequence.sequence_event_bus,
        parent=sequence,
    )
    sequence.resource_lifecycle_controller = lifecycle
    recipients = {
        name: Recipient(name)
        for name in (
            "trigger",
            "analysis-transport",
            "analysis",
            "workflow",
            "recording",
            "export",
        )
    }
    sequence.trigger_controller = recipients["trigger"]
    sequence.analysis_transport_controller = recipients["analysis-transport"]
    sequence.analysis_controller = recipients["analysis"]
    sequence.workflow_controller = recipients["workflow"]
    sequence.recording_controller = recipients["recording"]
    sequence.export_controller = recipients["export"]
    SequenceWindow._register_resource_lifecycle_recipients(sequence)

    sip.delete(sequence.analysis_controller)
    assert lifecycle._publish_lifecycle_request(34, "disconnect-domains") is True
    assert recipients["analysis"].calls == []
    assert sequence.sequence_event_bus.pending_resource_lifecycle_request_count == 0

    sequence.analysis_controller = Recipient("analysis-replacement")
    failed = Recipient("workflow-failed", False)
    sequence.workflow_controller = failed
    SequenceWindow._register_resource_lifecycle_recipients(sequence)
    assert lifecycle._publish_lifecycle_request(35, "disconnect-domains") is False
    assert failed.calls == [35]

    sip.delete(failed)
    replacement = Recipient("workflow-replacement")
    sequence.workflow_controller = replacement
    SequenceWindow._register_resource_lifecycle_recipients(sequence)
    assert lifecycle._publish_lifecycle_request(35, "disconnect-domains") is True
    assert replacement.calls == []
    assert lifecycle._publish_lifecycle_request(36, "disconnect-domains") is True
    assert replacement.calls == [36]
    assert sequence.sequence_event_bus.pending_resource_lifecycle_request_count == 0


def test_lifecycle_owner_finishes_domains_then_closes_dispatchers_after_ready_ack(
    qapp,
):
    widget = QWidget()
    bus = SequenceEventBus(widget)
    lifecycle = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(widget),
        SequenceResourceLifecycleModel(),
        lifecycle_bus=bus,
        parent=widget,
    )
    order = []
    domain_ready = {"value": False}

    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        lambda request: order.append(("workflow", request)) or True,
    )
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "export",
        lambda request: order.append(("export", request))
        or domain_ready["value"],
    )
    lifecycle._shutdown_prepared_generation = 23
    assert lifecycle.finalize_application_shutdown(23)

    assert lifecycle.complete_application_shutdown_delivery(23) is False
    assert [name for name, _request in order] == ["workflow", "export"]
    domain_ready["value"] = True
    assert lifecycle.complete_application_shutdown_delivery(23) is True
    assert [name for name, _request in order] == [
        "workflow",
        "export",
        "export",
    ]
    assert all(
        type(request) is ResourceLifecycleRequested
        and request.shutdown_generation == 23
        and request.operation == "disconnect-domains"
        for _name, request in order
    )

    assert lifecycle.complete_application_shutdown_after_ready_ack(23) is True
    assert [name for name, _request in order] == [
        "workflow",
        "export",
        "export",
    ]
    assert bus.pending_workflow_continuation_delivery_count == 0
    assert bus.pending_resource_lifecycle_request_count == 0


def test_hidden_lifecycle_suspend_never_publishes_domain_disconnect(qapp):
    widget = QWidget()

    class Publisher:
        def __init__(self):
            self.requests = []

        def publish_resource_lifecycle(self, request):
            self.requests.append(request)
            return True

    publisher = Publisher()
    lifecycle = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(widget),
        SequenceResourceLifecycleModel(),
        lifecycle_bus=publisher,
        parent=widget,
    )
    lifecycle._suspend_reusable_child_resources = lambda: True

    assert lifecycle.lightweight_child_cleanup() is True
    assert publisher.requests == []


def test_hidden_suspend_does_not_retire_production_registration(qapp):
    bus = SequenceEventBus()
    model = SequenceResourceLifecycleModel()
    owner = QObject()
    calls = []
    token = bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "recording",
        lambda request: calls.append(request) or True,
        owner=owner,
    )
    assert model.retain_resource_lifecycle_registration(
        "disconnect-domains", "recording", token, owner
    )
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(), model, lifecycle_bus=bus
    )
    lifecycle._suspend_reusable_child_resources = lambda: True

    assert lifecycle.lightweight_child_cleanup() is True
    assert model.resource_lifecycle_registrations[0].owner() is owner
    assert calls == []
    assert bus.pending_resource_lifecycle_request_count == 0


class _CloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class _HostileStateError(BaseException):
    def __str__(self):
        raise RuntimeError("diagnostics must not stringify this error")


class _ResourceLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


class _ShutdownFacade:
    def __init__(self, *, active=False):
        self.active = active
        self.requests = []
        self.raised = []

    def is_workflow_active(self):
        return self.active

    def request_application_shutdown(self, generation):
        self.requests.append(generation)
        return True

    def raise_shutdown_progress(self, generation):
        self.raised.append(generation)
        return True


class _MainHarness:
    def __init__(self, facade):
        self.sequence_window = facade
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _invoke_main(method_name, harness, *args):
    from main_window import MainWindow

    return getattr(MainWindow, method_name)(harness, *args)


def test_main_close_ignores_first_and_accepts_only_matching_ready_generation():
    facade = _ShutdownFacade()
    window = _MainHarness(facade)
    first = _CloseEvent()

    _invoke_main("closeEvent", window, first)

    assert first.ignored is True
    assert first.accepted is False
    assert facade.requests == [0]
    _invoke_main("on_shutdown_ready", window, ShutdownReady(99))
    assert window.close_calls == 0
    _invoke_main("on_shutdown_ready", window, ShutdownReady(0))
    assert window.close_calls == 1

    second = _CloseEvent()
    _invoke_main("closeEvent", window, second)
    assert second.accepted is True
    assert second.ignored is False

    third = _CloseEvent()
    _invoke_main("closeEvent", window, third)
    assert third.ignored is True
    assert third.accepted is False


def test_repeated_main_close_coalesces_one_generation_and_raises_progress():
    facade = _ShutdownFacade(active=True)
    window = _MainHarness(facade)

    first = _CloseEvent()
    repeated = _CloseEvent()
    _invoke_main("closeEvent", window, first)
    _invoke_main("closeEvent", window, repeated)

    assert first.ignored and repeated.ignored
    assert facade.requests == [0]
    assert facade.raised == [0]


def test_aborted_shutdown_clears_only_matching_main_generation():
    facade = _ShutdownFacade(active=True)
    window = _MainHarness(facade)
    _invoke_main("closeEvent", window, _CloseEvent())

    _invoke_main("on_shutdown_aborted", window, ShutdownAborted(7))
    _invoke_main("closeEvent", window, _CloseEvent())
    assert facade.requests == [0]

    _invoke_main("on_shutdown_aborted", window, ShutdownAborted(0))
    _invoke_main("closeEvent", window, _CloseEvent())
    assert facade.requests == [0, 1]


def test_destroyed_sequence_widget_during_first_close_is_ignored_without_exception():
    class DestroyedFacade:
        def request_application_shutdown(self, _generation):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    window = _MainHarness(DestroyedFacade())
    event = _CloseEvent()

    _invoke_main("closeEvent", window, event)

    assert event.ignored is True
    assert window._shutdown_active_generation is None


def test_guarded_second_close_accepts_without_a_third_close_handshake():
    facade = _ShutdownFacade()
    window = _MainHarness(facade)
    _invoke_main("closeEvent", window, _CloseEvent())
    _invoke_main("on_shutdown_ready", window, ShutdownReady(0))

    accepted = _CloseEvent()
    _invoke_main("closeEvent", window, accepted)
    assert accepted.accepted and not accepted.ignored
    assert window._shutdown_active_generation is None


def test_ready_guard_rejects_stale_shape_and_accepts_one_matching_close():
    facade = _ShutdownFacade()
    window = _MainHarness(facade)
    _invoke_main("_initialize_application_shutdown_state", window)
    window._shutdown_active_generation = 4

    assert _invoke_main(
        "on_shutdown_ready",
        window,
        SimpleNamespace(shutdown_generation=4),
    ) is False
    assert _invoke_main(
        "on_shutdown_ready", window, ShutdownReady(3)
    ) is False
    assert _invoke_main(
        "on_shutdown_ready", window, ShutdownReady(4)
    ) is True
    assert window.close_calls == 1

    second = _CloseEvent()
    _invoke_main("closeEvent", window, second)
    assert second.accepted and window._shutdown_active_generation is None


class _FatalReceiverError(BaseException):
    pass


@pytest.mark.parametrize(
    "error", [RuntimeError("deleted"), _FatalReceiverError("fatal"), SystemExit(4)]
)
def test_main_ready_receiver_contains_close_baseexceptions_for_dispatch_retry(error):
    facade = _ShutdownFacade()

    class RaisingMain(_MainHarness):
        def close(self):
            raise error

    window = RaisingMain(facade)
    _invoke_main("closeEvent", window, _CloseEvent())

    assert _invoke_main("on_shutdown_ready", window, ShutdownReady(0)) is False
    assert window._shutdown_close_permission_generation == 0


class _ShutdownView:
    def __init__(self):
        self.confirmations = []
        self.finished = []
        self.raised = []
        self.waiting = []

    def show_shutdown_confirmation(self, generation):
        self.confirmations.append(generation)
        return True

    def finish_shutdown_confirmation(self, generation):
        self.finished.append(generation)
        return True

    def raise_shutdown(self, generation):
        self.raised.append(generation)
        return True

    def show_shutdown_waiting(self, generation):
        self.waiting.append(generation)
        return True

    def finish_shutdown_waiting(self, generation):
        if generation in self.waiting:
            self.waiting.remove(generation)
        return True


def test_shutdown_coordinator_prompts_active_work_but_not_idle_awaiting_label(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    view = _ShutdownView()
    coordinator = SequenceShutdownCoordinator(model, bus, view=view)

    model.phase = WorkflowPhase.ANALYZING
    model.active_analysis_id = "analysis-1"
    model.shutdown_generation = 4
    model.shutdown_pending = True
    model.shutdown_asserted_active = True
    coordinator.synchronize()
    assert view.confirmations == [4]

    coordinator.disconnect()
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.retained_record_id = "record-1"
    model.awaiting_label = True
    model.shutdown_generation = 5
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    coordinator = SequenceShutdownCoordinator(model, bus, view=_ShutdownView())
    coordinator.synchronize()
    assert bus.commands.begin_shutdown_flush_requested is not None


def test_queued_shutdown_request_schedules_active_confirmation_without_polling(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel(workflow_generation=2)
    model.phase = WorkflowPhase.ANALYZING
    model.active_analysis_id = "analysis-1"
    SequenceWorkflowController(model, bus)
    view = _ShutdownView()
    coordinator = SequenceShutdownCoordinator(model, bus, view=view)

    assert coordinator.request_shutdown(17, True)
    qapp.processEvents()
    qapp.processEvents()

    assert model.shutdown_generation == 17
    assert view.confirmations == [17]


def test_confirmed_cancellation_replaces_decision_with_one_waiting_view():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.phase = WorkflowPhase.ANALYZING
    model.active_analysis_id = "analysis-1"
    model.shutdown_generation = 18
    model.shutdown_pending = True
    model.shutdown_asserted_active = True
    view = _ShutdownView()
    coordinator = SequenceShutdownCoordinator(model, bus, view=view)
    coordinator.synchronize()
    assert view.confirmations == [18]

    model.phase = WorkflowPhase.CANCELLING
    model.cancelling_phase = WorkflowPhase.ANALYZING
    model.cancelling_domain = "analysis"
    model.shutdown_cancellation_confirmed = True
    coordinator.synchronize()
    coordinator.synchronize()

    assert view.finished == [18]
    assert view.waiting == [18]


def test_shutdown_coordinator_runs_cleanup_before_one_ready_and_ignores_stale(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 3
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    order = []
    ready = []
    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: order.append(("cleanup", generation)) or True,
    )
    bus.register_workflow_continuation_recipient(
        "shutdown-ready",
        "main",
        lambda event: ready.append(event) or order == [("cleanup", 3)],
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(2)) is False
    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(3)) is True
    assert order == [("cleanup", 3)]
    assert len(ready) == 1
    assert ready[0] == ShutdownReady(3)
    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(3)) is False
    assert len(ready) == 1


@pytest.mark.parametrize(
    ("status", "acknowledged"),
    [
        (WorkflowContinuationDeliveryStatus.ACK, True),
        (WorkflowContinuationDeliveryStatus.RETRYABLE_NACK, False),
        (WorkflowContinuationDeliveryStatus.PERMANENT_REJECT, False),
    ],
)
def test_shutdown_detailed_compatibility_ready_preserves_boolean_projection(
    qapp, status, acknowledged
):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 34
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    calls = []
    legacy_calls = []
    cleanup = []
    finalized = []
    released = []
    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: cleanup.append(generation) or True,
        finalize_after_ready_ack=(
            lambda generation: finalized.append(generation) or True
        ),
        release_shutdown_close=lambda generation: released.append(generation) or True,
    )

    def deliver_outcome(delivery_id, kind, message, *, owner):
        calls.append((delivery_id, kind, message, owner))
        return WorkflowContinuationDeliveryOutcome(status, "compatibility outcome")

    def legacy_delivery(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return status is WorkflowContinuationDeliveryStatus.ACK

    bus.deliver_workflow_continuation_outcome = deliver_outcome
    bus.deliver_workflow_continuation = legacy_delivery

    result = coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(34))

    assert result is acknowledged
    assert calls == [
        (
            ("shutdown-ready", 34),
            "shutdown-ready",
            ShutdownReady(34),
            coordinator,
        )
    ]
    assert legacy_calls == []
    assert cleanup == [34]
    assert finalized == ([34] if acknowledged else [])
    assert released == ([34] if acknowledged else [])
    assert coordinator._completed_generations == ({34} if acknowledged else set())
    assert coordinator._ready_ack_generation == (34 if acknowledged else None)
    if not acknowledged:
        assert coordinator._ready_pending_generation == 34
        assert coordinator._active is True
        coordinator._retry_timer.stop()


def test_cleanup_and_ready_delivery_retry_without_sealing_generation(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 41
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    cleanup_results = [KeyboardInterrupt(), False, True]
    deliveries = []

    def cleanup(_generation):
        result = cleanup_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    coordinator = SequenceShutdownCoordinator(
        model, bus, view=_ShutdownView(), cleanup_resources=cleanup
    )

    def main_ready(event):
        deliveries.append(event)
        if len(deliveries) == 1:
            raise SystemExit("retry ready")
        return True

    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "main", main_ready
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(41)) is False
    assert coordinator._completed_generations == set()
    assert coordinator.retry_pending_shutdown() is False
    assert coordinator._completed_generations == set()
    assert coordinator.retry_pending_shutdown() is False
    assert deliveries == [ShutdownReady(41)]
    assert coordinator._completed_generations == set()
    assert coordinator.retry_pending_shutdown() is True
    assert deliveries == [ShutdownReady(41), ShutdownReady(41)]
    assert coordinator._completed_generations == {41}


def test_partial_ready_delivery_does_not_complete_generation_before_all_acks():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 56
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    coordinator = SequenceShutdownCoordinator(model, bus, view=_ShutdownView())
    observer_calls = []

    def main(_event):
        return True

    def observer(_event):
        observer_calls.append(True)
        return len(observer_calls) > 1

    bus.register_workflow_continuation_recipient("shutdown-ready", "main", main)
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "observer", observer
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(56)) is False
    assert coordinator._completed_generations == set()
    assert bus.pending_workflow_continuation_delivery_count == 1
    assert coordinator.retry_pending_shutdown() is True
    assert coordinator._completed_generations == {56}


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit(57)],
)
def test_resource_finalization_failure_prevents_ready_until_exact_retry(failure):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 57
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    outcomes = [failure, True]
    finalized = []

    def finalize(generation):
        finalized.append(generation)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=finalize,
    )
    ready = []
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "main", lambda event: ready.append(event) or True
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(57)) is False
    assert coordinator._completed_generations == set()
    assert ready == []
    assert coordinator.retry_pending_shutdown() is True
    assert coordinator._completed_generations == {57}
    assert finalized == [57, 57]
    assert ready == [ShutdownReady(57)]


def test_shutdown_finalization_reentry_coalesces_one_single_flight_attempt(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 59
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    cleanup_calls = []
    reentered = []
    continuation_queued = []
    holder = {}

    def cleanup(generation):
        cleanup_calls.append(generation)
        reentered.append(holder["coordinator"].retry_pending_shutdown())
        continuation_queued.append(holder["coordinator"]._retry_timer.isActive())
        return True

    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=cleanup,
    )
    holder["coordinator"] = coordinator
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "main", lambda _event: True
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(59))
    qapp.processEvents()
    assert cleanup_calls == [59]
    assert reentered == [True]
    assert continuation_queued == [True]
    assert coordinator._completed_generations == {59}


def test_shutdown_ready_ack_finalizes_internal_owners_before_close_release():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 60
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    order = []
    holder = {}

    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: order.append(("cleanup", generation)) or True,
        finalize_after_ready_ack=(
            lambda generation: order.append(("finalize", generation)) or True
        ),
        release_shutdown_close=lambda generation: (
            order.append(("release", generation, holder["coordinator"]._active))
            or True
        ),
    )
    holder["coordinator"] = coordinator
    bus.register_workflow_continuation_recipient(
        "shutdown-ready",
        "main-stage",
        lambda event: order.append(("ready-ack", event.shutdown_generation)) or True,
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(60))
    assert order == [
        ("cleanup", 60),
        ("ready-ack", 60),
        ("finalize", 60),
        ("release", 60, False),
    ]
    assert coordinator._active is False


def test_shutdown_internal_callback_owner_is_collectible_after_ready_release():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 60
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING

    class Ports(QObject):
        def cleanup(self, _generation):
            return True

        def finalize(self, _generation):
            return True

        def release(self, _generation):
            return True

    ports = Ports()
    ports_ref = weakref.ref(ports)
    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=ports.cleanup,
        finalize_after_ready_ack=ports.finalize,
        release_shutdown_close=ports.release,
    )
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "stage", lambda _event: True
    )

    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(60))
    del ports
    gc.collect()
    assert ports_ref() is None


def test_production_ready_stage_closes_dispatchers_before_main_release(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    sequence = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(sequence)
    sequence.sequence_event_bus = SequenceEventBus(sequence)
    sequence.workflow_model = SequenceWorkflowModel()
    sequence.workflow_model.shutdown_generation = 60
    sequence.workflow_model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    SequenceWindow._initialize_shutdown_ready_release_port(sequence)
    observations = []

    class MainRecipient(QObject):
        def ready(self, event):
            observations.append(
                (
                    event,
                    sequence.shutdown_coordinator._active,
                    sequence.sequence_event_bus._workflow_continuation_dispatch_active,
                    sequence.sequence_event_bus._resource_lifecycle_dispatch_active,
                    sequence.sequence_event_bus.pending_workflow_continuation_delivery_count,
                    sequence.sequence_event_bus.pending_resource_lifecycle_request_count,
                )
            )
            return True

    main = MainRecipient()
    main_ref = weakref.ref(main)
    assert sequence.register_shutdown_ready_recipient(main.ready, owner=main)

    def finalize_after_ack(_generation):
        sequence.sequence_event_bus.close_workflow_continuation_dispatcher()
        sequence.sequence_event_bus.close_resource_lifecycle_dispatcher()
        return True

    sequence.shutdown_coordinator = SequenceShutdownCoordinator(
        sequence.workflow_model,
        sequence.sequence_event_bus,
        view=_ShutdownView(),
        cleanup_resources=lambda _generation: True,
        finalize_after_ready_ack=finalize_after_ack,
        release_shutdown_close=sequence._release_staged_shutdown_close,
        parent=sequence,
    )

    assert sequence.shutdown_coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(60)
    )
    assert observations == [(ShutdownReady(60), False, False, False, 0, 0)]
    assert sequence.sequence_event_bus.workflow_continuation_lifecycle_owner_count == 0
    del main
    gc.collect()
    assert main_ref() is None


def test_repeated_top_level_close_restarts_exhausted_staged_release_round(qapp):
    from main_window import MainWindow
    from ui.sequence.sequence_widget import SequenceWindow

    sequence = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(sequence)
    sequence.sequence_event_bus = SequenceEventBus(sequence)
    sequence.workflow_model = SequenceWorkflowModel()
    sequence.workflow_model.shutdown_generation = 63
    sequence.workflow_model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    SequenceWindow._initialize_shutdown_ready_release_port(sequence)
    outcomes = [
        False,
        RuntimeError("ordinary"),
        KeyboardInterrupt(),
        SystemExit(9),
        False,
        False,
        True,
    ]
    finalizations = []
    accepted_close_events = []

    class TopLevel(QObject):
        def __init__(self):
            super().__init__()
            self.sequence_window = sequence
            self._shutdown_generation_counter = 63
            self._shutdown_active_generation = 63
            self._shutdown_close_permission_generation = None

        def close(self):
            event = _CloseEvent()
            MainWindow.closeEvent(self, event)
            accepted_close_events.append(event.accepted)
            return event.accepted

    top_level = TopLevel()

    class ReadyRecipient(QObject):
        def __init__(self):
            super().__init__()
            self.events = []

        def ready(self, event):
            self.events.append(event)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is False:
                return False
            return MainWindow.on_shutdown_ready(top_level, event)

    recipient = ReadyRecipient()
    assert sequence.register_shutdown_ready_recipient(
        recipient.ready, owner=recipient
    )

    def finalize_after_ack(_generation):
        finalizations.append(63)
        sequence.sequence_event_bus.close_workflow_continuation_dispatcher()
        sequence.sequence_event_bus.close_resource_lifecycle_dispatcher()
        return True

    sequence.shutdown_coordinator = SequenceShutdownCoordinator(
        sequence.workflow_model,
        sequence.sequence_event_bus,
        view=_ShutdownView(),
        cleanup_resources=lambda _generation: True,
        finalize_after_ready_ack=finalize_after_ack,
        release_shutdown_close=sequence._release_staged_shutdown_close,
        parent=sequence,
    )

    assert sequence.shutdown_coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(63)
    )
    QTest.qWait(900)
    assert len(recipient.events) == 6
    assert sequence._shutdown_close_release_pending_generation == 63
    assert not sequence._shutdown_close_release_timer.isActive()
    assert sequence.shutdown_coordinator._active is False
    assert finalizations == [63]

    inactive_coordinator_calls = []
    sequence.shutdown_coordinator.raise_progress = (
        lambda generation: inactive_coordinator_calls.append(generation) or False
    )
    assert sequence.raise_shutdown_progress(62) is False
    assert not sequence._shutdown_close_release_timer.isActive()

    repeated_close = _CloseEvent()
    MainWindow.closeEvent(top_level, repeated_close)
    assert repeated_close.ignored
    assert sequence._shutdown_close_release_timer.isActive()
    coalesced_close = _CloseEvent()
    MainWindow.closeEvent(top_level, coalesced_close)
    assert coalesced_close.ignored
    assert sequence._shutdown_close_release_timer.isActive()
    assert inactive_coordinator_calls == []

    qapp.processEvents()
    assert len(recipient.events) == 7
    assert accepted_close_events == [True]
    assert top_level._shutdown_active_generation is None
    assert sequence._shutdown_close_release_pending_generation is None
    assert sequence._shutdown_ready_staged_event is None
    assert sequence._shutdown_close_release_attempt_generation is None
    assert sequence._shutdown_close_release_attempt_token is None
    assert sequence._shutdown_close_release_retry_queued_generation is None
    assert not sequence._shutdown_close_release_timer.isActive()
    assert sequence.sequence_event_bus.pending_workflow_continuation_delivery_count == 0
    assert sequence.sequence_event_bus.pending_resource_lifecycle_request_count == 0
    assert finalizations == [63]


def test_reentrant_staged_attempt_is_not_a_main_close_ack(qapp):
    from main_window import MainWindow
    from ui.sequence.sequence_widget import SequenceWindow

    sequence = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(sequence)
    sequence.sequence_event_bus = SequenceEventBus(sequence)
    sequence.workflow_model = SequenceWorkflowModel()
    sequence.workflow_model.shutdown_generation = 64
    sequence.workflow_model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    SequenceWindow._initialize_shutdown_ready_release_port(sequence)
    finalizations = []
    accepted_close_events = []
    nested_attempt_results = []

    class TopLevel(QObject):
        def __init__(self):
            super().__init__()
            self.sequence_window = sequence
            self._shutdown_generation_counter = 64
            self._shutdown_active_generation = 64
            self._shutdown_close_permission_generation = None

        def close(self):
            event = _CloseEvent()
            MainWindow.closeEvent(self, event)
            accepted_close_events.append(event.accepted)
            return event.accepted

    top_level = TopLevel()

    class ReentrantRecipient(QObject):
        def __init__(self):
            super().__init__()
            self.events = []

        def ready(self, event):
            self.events.append(event)
            attempt = len(self.events)
            if attempt == 1:
                result = sequence._attempt_staged_shutdown_close_release()
                nested_attempt_results.append(result)
                return result
            if attempt == 2:
                return None
            if attempt == 3:
                raise KeyboardInterrupt()
            return MainWindow.on_shutdown_ready(top_level, event)

    recipient = ReentrantRecipient()
    assert sequence.register_shutdown_ready_recipient(
        recipient.ready, owner=recipient
    )

    def finalize_after_ack(_generation):
        finalizations.append(64)
        sequence.sequence_event_bus.close_workflow_continuation_dispatcher()
        sequence.sequence_event_bus.close_resource_lifecycle_dispatcher()
        return True

    sequence.shutdown_coordinator = SequenceShutdownCoordinator(
        sequence.workflow_model,
        sequence.sequence_event_bus,
        view=_ShutdownView(),
        cleanup_resources=lambda _generation: True,
        finalize_after_ready_ack=finalize_after_ack,
        release_shutdown_close=sequence._release_staged_shutdown_close,
        parent=sequence,
    )

    assert sequence.shutdown_coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(64)
    )
    assert nested_attempt_results == [False]
    assert sequence._shutdown_close_release_pending_generation == 64
    assert sequence._shutdown_close_release_timer.isActive()
    assert finalizations == [64]

    qapp.processEvents()
    assert len(recipient.events) == 2
    assert sequence._shutdown_close_release_pending_generation == 64
    assert sequence._shutdown_close_release_timer.isActive()

    sequence._shutdown_close_release_timer.stop()
    assert sequence._attempt_staged_shutdown_close_release() is False
    assert len(recipient.events) == 3
    assert sequence._shutdown_close_release_pending_generation == 64
    assert sequence._shutdown_close_release_timer.isActive()

    inactive_coordinator_calls = []
    sequence.shutdown_coordinator.raise_progress = (
        lambda generation: inactive_coordinator_calls.append(generation) or False
    )
    first_close = _CloseEvent()
    second_close = _CloseEvent()
    MainWindow.closeEvent(top_level, first_close)
    MainWindow.closeEvent(top_level, second_close)
    assert first_close.ignored and second_close.ignored
    assert sequence._shutdown_close_release_timer.isActive()
    assert inactive_coordinator_calls == []

    qapp.processEvents()
    assert len(recipient.events) == 4
    assert accepted_close_events == [True]
    assert top_level._shutdown_active_generation is None
    assert sequence._shutdown_close_release_pending_generation is None
    assert sequence._shutdown_ready_staged_event is None
    assert sequence._shutdown_close_release_attempt_token is None
    assert sequence._shutdown_close_release_retry_queued_generation is None
    assert not sequence._shutdown_close_release_timer.isActive()
    assert sequence.sequence_event_bus.pending_workflow_continuation_delivery_count == 0
    assert sequence.sequence_event_bus.pending_resource_lifecycle_request_count == 0
    assert finalizations == [64]


@pytest.mark.parametrize("stage", ["cleanup", "ready-recipient"])
def test_shutdown_native_deletion_during_external_callback_is_terminal_safe(qapp, stage):
    from PyQt5 import sip

    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 61
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    holder = {}

    def cleanup(_generation):
        if stage == "cleanup":
            sip.delete(holder["coordinator"])
        return True

    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=cleanup,
    )
    holder["coordinator"] = coordinator

    def ready(_event):
        if stage == "ready-recipient":
            sip.delete(coordinator)
        return True

    bus.register_workflow_continuation_recipient("shutdown-ready", "main", ready)

    result = coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(61))
    assert result is False
    assert sip.isdeleted(coordinator)
    qapp.processEvents()


def test_cleanup_retry_uses_bounded_backoff_and_repeated_close_restarts_round(qapp):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 58
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    cleanup_calls = []

    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: cleanup_calls.append(generation) or False,
    )
    assert coordinator.handle_shutdown_flush_completed(ShutdownFlushCompleted(58)) is False
    QTest.qWait(1_200)
    first_round = len(cleanup_calls)
    assert first_round <= 6
    assert coordinator._retry_timer.isActive() is False

    coordinator.raise_progress(58)
    QTest.qWait(60)
    assert len(cleanup_calls) > first_round


def test_failed_confirmation_presentation_is_retryable_by_repeated_close():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 73
    model.shutdown_pending = True
    model.shutdown_asserted_active = True
    model.phase = WorkflowPhase.ANALYZING

    class FlakyView(_ShutdownView):
        def show_shutdown_confirmation(self, generation):
            self.confirmations.append(generation)
            return len(self.confirmations) > 1

        def raise_shutdown(self, generation):
            self.raised.append(generation)
            return False

    view = FlakyView()
    coordinator = SequenceShutdownCoordinator(model, bus, view=view)
    assert coordinator.synchronize() is False
    assert coordinator.raise_progress(73) is True
    assert view.confirmations == [73, 73]


@pytest.mark.parametrize(
    "phase, identifier_attr, identifier, cancel_signal",
    [
        (WorkflowPhase.RECORDING, "active_session_id", "session-1", "cancel_recording_requested"),
        (WorkflowPhase.ANALYZING, "active_analysis_id", "analysis-1", "cancel_analysis_requested"),
        (WorkflowPhase.RESULT_EXPORTING, "active_job_id", "job-1", "cancel_export_requested"),
    ],
)
def test_confirmed_shutdown_posts_exactly_one_domain_cancel_and_waits_for_terminal(
    phase, identifier_attr, identifier, cancel_signal
):
    bus = SequenceEventBus()
    model = SequenceWorkflowModel(workflow_generation=7)
    model.phase = phase
    setattr(model, identifier_attr, identifier)
    if phase is WorkflowPhase.RECORDING:
        model.active_session_origin = SessionOrigin.CANONICAL
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    cancellations = []
    getattr(bus.commands, cancel_signal).connect(cancellations.append)

    assert controller.handle_shutdown(ShutdownRequested(21, True)) is True
    assert model.phase is phase
    assert model.shutdown_pending is True
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(21)
    ) is True
    assert model.phase is WorkflowPhase.CANCELLING
    assert len(cancellations) == 1
    assert controller.handle_confirm_shutdown_cancellation(
        ConfirmShutdownCancellationRequested(21)
    ) is True
    assert len(cancellations) == 1


def test_cancelling_close_returns_to_running_and_awaiting_label_skips_confirmation():
    from ui.sequence.sequence_messages import AbortShutdownRequested

    bus = SequenceEventBus()
    model = SequenceWorkflowModel(workflow_generation=3)
    model.phase = WorkflowPhase.ANALYZING
    model.active_analysis_id = "analysis-1"
    controller = SequenceWorkflowController(model, bus, connect_bus=False)
    aborted = []
    bus.events.shutdown_aborted.connect(aborted.append)
    assert controller.handle_shutdown(ShutdownRequested(5, True))
    assert controller.handle_abort_shutdown(AbortShutdownRequested(5))
    assert model.phase is WorkflowPhase.ANALYZING
    assert model.shutdown_generation is None
    assert aborted == [ShutdownAborted(5)]

    idle = SequenceWorkflowModel()
    idle.retained_record_id = "record-1"
    idle.awaiting_label = True
    idle_controller = SequenceWorkflowController(idle, bus, connect_bus=False)
    assert idle_controller.handle_shutdown(ShutdownRequested(6, False))
    assert idle.phase is WorkflowPhase.SHUTDOWN_FLUSHING
    assert idle.shutdown_pending is False


class _ExportView:
    def __init__(self):
        self.shutdown_failures = []
        self.finished = []
        self.ordinary_failures = []

    def show_progress(self, *_args):
        return True

    def finish(self, *_args):
        return True

    def show_shutdown_failure(self, generation, job_id, attempt_id, failures):
        self.shutdown_failures.append((generation, job_id, attempt_id, tuple(failures)))
        return True

    def show_failure(self, job_id, attempt_id, failures):
        self.ordinary_failures.append((job_id, attempt_id, tuple(failures)))
        return True

    def finish_shutdown_failure(self, generation, job_id, attempt_id):
        self.finished.append((generation, job_id, attempt_id))
        return True

    def disconnect(self):
        return None


@pytest.mark.parametrize(
    ("status", "acknowledged"),
    [
        (WorkflowContinuationDeliveryStatus.ACK, True),
        (WorkflowContinuationDeliveryStatus.RETRYABLE_NACK, False),
        (WorkflowContinuationDeliveryStatus.PERMANENT_REJECT, False),
    ],
)
def test_shutdown_detailed_compatibility_flush_completed_preserves_boolean_projection(
    qapp, status, acknowledged
):
    bus = SequenceEventBus()
    model = SequenceExportModel()
    controller = SequenceExportController(
        model,
        _ExportView(),
        bus=bus,
        submit_attempt=lambda *_args: None,
    )
    calls = []
    legacy_calls = []

    def deliver_outcome(delivery_id, kind, message, *, owner):
        calls.append((delivery_id, kind, message, owner))
        return WorkflowContinuationDeliveryOutcome(status, "compatibility outcome")

    def legacy_delivery(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return status is WorkflowContinuationDeliveryStatus.ACK

    bus.deliver_workflow_continuation_outcome = deliver_outcome
    bus.deliver_workflow_continuation = legacy_delivery
    model.shutdown_flush_pending = True
    model.shutdown_flush_generation = 35

    result = controller._publish_shutdown_flush_completed(35)

    assert result is acknowledged
    assert calls == [
        (
            ("shutdown-flush-completed", 35, "shutdown:35", "attempt:0"),
            "shutdown-flush-completed",
            ShutdownFlushCompleted(35),
            controller,
        )
    ]
    assert legacy_calls == []
    assert model.shutdown_flush_terminal is acknowledged
    assert model.shutdown_flush_pending is not acknowledged
    assert controller._shutdown_completion_retry_attempt == (0 if acknowledged else 1)
    if not acknowledged:
        controller._shutdown_completion_retry_timer.stop()


def test_export_shutdown_with_no_targets_completes_once_and_stale_begin_is_rejected():
    bus = SequenceEventBus()
    completed = []
    bus.events.shutdown_flush_completed.connect(completed.append)
    controller = SequenceExportController(
        SequenceExportModel(),
        _ExportView(),
        bus=bus,
        submit_attempt=lambda *_args: None,
        connect_bus=False,
    )

    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(8)) is True
    assert completed == [ShutdownFlushCompleted(8)]
    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(8)) is True
    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(9)) is False
    assert completed == [ShutdownFlushCompleted(8)]


def _shutdown_export_with_target(tmp_path):
    bus = SequenceEventBus()
    view = _ExportView()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    output = tmp_path / "daily.xlsx"
    spool = tmp_path / "daily.csv"
    output.write_bytes(b"existing-xlsx")
    spool.write_bytes(b"existing-spool")
    target = SpoolTarget.create("Daily", {}, str(output), str(spool))
    controller.schedule_spool_targets((target,))
    completed = []
    bus.events.shutdown_flush_completed.connect(completed.append)
    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(31))
    assert len(submissions) == 1
    return controller, view, submissions, completed, output, spool


def test_shutdown_flush_failure_retry_uses_new_attempt_and_stale_terminal_is_ignored(tmp_path):
    controller, view, submissions, completed, _output, _spool = (
        _shutdown_export_with_target(tmp_path)
    )
    first, _attempt = submissions.pop()
    failed = SimpleNamespace(
        job_id=first.job_id,
        attempt_id=first.attempt_id,
        failures=("locked",),
    )
    assert controller.handle_worker_failed(failed)
    assert view.shutdown_failures[-1][:3] == (
        31,
        first.job_id,
        first.attempt_id,
    )

    from ui.sequence.sequence_messages import RetryShutdownFlushRequested

    assert controller.handle_retry_shutdown_flush(
        RetryShutdownFlushRequested(31, first.job_id, first.attempt_id)
    )
    retry, _attempt = submissions.pop()
    assert retry.job_id == first.job_id
    assert retry.attempt_id != first.attempt_id
    assert controller.handle_worker_completed(
        SimpleNamespace(job_id=first.job_id, attempt_id=first.attempt_id)
    ) is False
    assert completed == []
    assert controller.handle_worker_completed(
        SimpleNamespace(job_id=retry.job_id, attempt_id=retry.attempt_id)
    )
    assert completed == [ShutdownFlushCompleted(31)]


@pytest.mark.parametrize(
    "error",
    [RuntimeError("submit failed"), KeyboardInterrupt(), SystemExit(9)],
)
def test_shutdown_submit_port_baseexceptions_use_exact_shutdown_failure(
    tmp_path, error
):
    bus = SequenceEventBus()
    view = _ExportView()

    def fail_submit(_job, _attempt):
        raise error

    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=fail_submit,
        connect_bus=False,
    )
    target = SpoolTarget.create(
        "Daily", {}, str(tmp_path / "daily.xlsx"), str(tmp_path / "daily.csv")
    )
    controller.schedule_spool_targets((target,))

    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(81))
    identity = controller.model.shutdown_flush_failure_identity
    assert identity is not None
    assert view.shutdown_failures[-1][:3] == (81, *identity)
    assert view.ordinary_failures == []


class _SetupSignal:
    def __init__(self, *, fail_connect=False):
        self.fail_connect = fail_connect

    def connect(self, _slot, *_args):
        if self.fail_connect:
            raise RuntimeError("connect failed")


class _SetupThread:
    def __init__(self, stage):
        self.stage = stage
        self.started = _SetupSignal(fail_connect=stage == "connect")
        self.finished = _SetupSignal()

    def start(self):
        if self.stage == "start":
            raise RuntimeError("start failed")

    def isRunning(self):
        return False

    def quit(self):
        return None

    def deleteLater(self):
        return None


class _SetupWorker:
    def __init__(self, stage):
        self.stage = stage
        self.completed = _SetupSignal()
        self.failed = _SetupSignal()
        self.finished = _SetupSignal()

    def moveToThread(self, _thread):
        if self.stage == "move":
            raise RuntimeError("move failed")

    def run(self):
        return None

    def deleteLater(self):
        return None


@pytest.mark.parametrize("stage", ["thread", "worker", "move", "connect", "start"])
def test_shutdown_worker_setup_boundaries_share_retryable_failure_state(tmp_path, stage):
    bus = SequenceEventBus()
    view = _ExportView()

    def thread_factory():
        if stage == "thread":
            raise RuntimeError("thread failed")
        return _SetupThread(stage)

    def worker_factory(*_args, **_kwargs):
        if stage == "worker":
            raise RuntimeError("worker failed")
        return _SetupWorker(stage)

    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        thread_factory=thread_factory,
        worker_factory=worker_factory,
        connect_bus=False,
    )
    target = SpoolTarget.create(
        "Daily", {}, str(tmp_path / "daily.xlsx"), str(tmp_path / "daily.csv")
    )
    controller.schedule_spool_targets((target,))

    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(82))
    identity = controller.model.shutdown_flush_failure_identity
    assert identity is not None
    assert view.shutdown_failures[-1][:3] == (82, *identity)
    assert view.ordinary_failures == []


@pytest.mark.parametrize("error", [RuntimeError("terminal"), KeyboardInterrupt(), SystemExit(3)])
def test_shutdown_terminal_handler_baseexceptions_keep_exact_attempt(tmp_path, error):
    controller, view, submissions, completed, _output, _spool = (
        _shutdown_export_with_target(tmp_path)
    )
    job, _attempt = submissions.pop()

    def fail_terminal(*_args, **_kwargs):
        raise error

    controller.model.complete_rebuild = fail_terminal
    assert controller.handle_worker_completed(
        SimpleNamespace(job_id=job.job_id, attempt_id=job.attempt_id)
    ) is False
    assert controller.model.shutdown_flush_failure_identity == (
        job.job_id,
        job.attempt_id,
    )
    assert view.shutdown_failures[-1][:3] == (
        31,
        job.job_id,
        job.attempt_id,
    )
    assert completed == []


def test_shutdown_failure_presentation_retries_same_identity_then_new_attempt(tmp_path):
    bus = SequenceEventBus()

    class FlakyExportView(_ExportView):
        def show_shutdown_failure(self, generation, job_id, attempt_id, failures):
            self.shutdown_failures.append((generation, job_id, attempt_id, tuple(failures)))
            return len(self.shutdown_failures) > 1

    view = FlakyExportView()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        view,
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    target = SpoolTarget.create(
        "Daily", {}, str(tmp_path / "daily.xlsx"), str(tmp_path / "daily.csv")
    )
    controller.schedule_spool_targets((target,))
    controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(83))
    first, _attempt = submissions.pop()
    controller.handle_worker_failed(
        SimpleNamespace(
            job_id=first.job_id,
            attempt_id=first.attempt_id,
            failures=("locked",),
        )
    )
    assert len(view.shutdown_failures) == 1
    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(83))
    assert len(view.shutdown_failures) == 2

    from ui.sequence.sequence_messages import RetryShutdownFlushRequested

    assert controller.handle_retry_shutdown_flush(
        RetryShutdownFlushRequested(83, first.job_id, first.attempt_id)
    )
    retry, _attempt = submissions.pop()
    assert retry.attempt_id != first.attempt_id


@pytest.mark.parametrize(
    "first_result",
    [False, RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit(5)],
)
def test_shutdown_flush_completed_retries_exact_formal_delivery_before_terminal(
    first_result, qapp
):
    bus = SequenceEventBus()
    calls = []

    def recipient(event):
        calls.append(event)
        if len(calls) == 1:
            if isinstance(first_result, BaseException):
                raise first_result
            return first_result
        return True

    bus.register_workflow_continuation_recipient(
        "shutdown-flush-completed", "coordinator", recipient
    )
    controller = SequenceExportController(
        SequenceExportModel(),
        _ExportView(),
        bus=bus,
        connect_bus=True,
    )

    assert controller.begin_shutdown_flush(84)
    assert controller.model.shutdown_flush_pending is True
    assert controller.model.shutdown_flush_terminal is False
    identity = controller.model.shutdown_flush_completion_identity
    assert identity is not None and identity[0] == 84
    assert calls == [ShutdownFlushCompleted(84)]

    assert controller.begin_shutdown_flush(84)
    assert calls == [ShutdownFlushCompleted(84), ShutdownFlushCompleted(84)]
    assert controller.model.shutdown_flush_pending is False
    assert controller.model.shutdown_flush_terminal is True
    assert controller.model.shutdown_flush_completion_identity == identity


def test_shutdown_flush_completed_waits_for_all_formal_recipients():
    bus = SequenceEventBus()
    observer_calls = []
    bus.register_workflow_continuation_recipient(
        "shutdown-flush-completed", "coordinator", lambda _event: True
    )
    bus.register_workflow_continuation_recipient(
        "shutdown-flush-completed",
        "critical-observer",
        lambda _event: observer_calls.append(True) or len(observer_calls) > 1,
    )
    controller = SequenceExportController(
        SequenceExportModel(), _ExportView(), bus=bus, connect_bus=True
    )

    assert controller.begin_shutdown_flush(85)
    assert controller.model.shutdown_flush_terminal is False
    assert controller.begin_shutdown_flush(85)
    assert controller.model.shutdown_flush_terminal is True
    assert observer_calls == [True, True]


def test_shutdown_flush_completed_automatic_retry_is_bounded_and_repeat_begin_restarts(qapp):
    bus = SequenceEventBus()
    calls = []
    bus.register_workflow_continuation_recipient(
        "shutdown-flush-completed",
        "coordinator",
        lambda event: calls.append(event) or False,
    )
    controller = SequenceExportController(
        SequenceExportModel(), _ExportView(), bus=bus, connect_bus=True
    )

    assert controller.begin_shutdown_flush(86)
    QTest.qWait(1_200)
    first_round = len(calls)
    assert first_round <= 6
    assert controller._shutdown_completion_retry_timer.isActive() is False
    assert controller.model.shutdown_flush_pending is True
    assert controller.model.shutdown_flush_terminal is False

    assert controller.begin_shutdown_flush(86)
    assert len(calls) > first_round


def test_shutdown_waits_for_active_rebuild_then_runs_one_final_dirty_generation(tmp_path):
    bus = SequenceEventBus()
    submissions = []
    controller = SequenceExportController(
        SequenceExportModel(),
        _ExportView(),
        bus=bus,
        submit_attempt=lambda job, attempt: submissions.append((job, attempt)),
        connect_bus=False,
    )
    target = SpoolTarget.create(
        "Daily", {}, str(tmp_path / "daily.xlsx"), str(tmp_path / "daily.csv")
    )
    controller.schedule_spool_targets((target,))
    controller.handle_rebuild_debounce()
    active, _attempt = submissions.pop()
    completed = []
    bus.events.shutdown_flush_completed.connect(completed.append)

    assert controller.handle_begin_shutdown_flush(BeginShutdownFlushRequested(32))
    assert completed == []
    assert controller.handle_worker_completed(
        SimpleNamespace(job_id=active.job_id, attempt_id=active.attempt_id)
    )
    assert completed == []
    assert len(submissions) == 1
    final, _attempt = submissions.pop()
    assert final.generation > active.generation
    assert controller.handle_worker_completed(
        SimpleNamespace(job_id=final.job_id, attempt_id=final.attempt_id)
    )
    assert completed == [ShutdownFlushCompleted(32)]


def test_shutdown_flush_ignore_keeps_existing_files_untouched_and_completes(tmp_path):
    controller, _view, submissions, completed, output, spool = (
        _shutdown_export_with_target(tmp_path)
    )
    job, _attempt = submissions.pop()
    assert controller.handle_worker_failed(
        SimpleNamespace(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            failures=("locked",),
        )
    )
    assert controller.handle_ignore_shutdown_flush_failure(
        IgnoreShutdownFlushFailureRequested(
            31, job.job_id, job.attempt_id
        )
    )
    assert output.read_bytes() == b"existing-xlsx"
    assert spool.read_bytes() == b"existing-spool"
    assert completed == [ShutdownFlushCompleted(31)]


def test_shutdown_ignore_requires_exact_generation_job_attempt_and_does_not_run_io():
    bus = SequenceEventBus()
    controller = SequenceExportController(
        SequenceExportModel(),
        _ExportView(),
        bus=bus,
        submit_attempt=lambda *_args: None,
        connect_bus=False,
    )
    controller.model.shutdown_flush_pending = True
    controller.model.shutdown_flush_generation = 12
    controller.model.shutdown_flush_failure_identity = ("job-1", "attempt-1")

    stale = IgnoreShutdownFlushFailureRequested(11, "job-1", "attempt-1")
    wrong = IgnoreShutdownFlushFailureRequested(12, "job-1", "attempt-2")
    assert controller.handle_ignore_shutdown_flush_failure(stale) is False
    assert controller.handle_ignore_shutdown_flush_failure(wrong) is False


def test_hidden_child_close_path_is_lightweight_and_visible_path_has_no_sync_loop():
    source = Path("ui/sequence/sequence_widget.py").read_text(encoding="utf-8")
    close_source = source[source.index("    def closeEvent(self, event):"):]
    close_source = close_source[: close_source.index("    def flush_excel_spool_build")]
    assert "while True" not in close_source
    assert "QApplication.processEvents" not in close_source
    assert ".exec_()" not in close_source
    assert "request_application_shutdown" in source
    assert "_lightweight_child_cleanup" in source


def test_hidden_reusable_child_suspends_resources_without_disconnecting_mvc(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Trigger:
        is_active = True
        model = SimpleNamespace(tcp_enabled=True)

        def stop_tcp(self):
            calls.append("tcp-stop")
            self.model.tcp_enabled = False
            return True

        def set_tcp_enabled(self, enabled):
            calls.append(("tcp-resume", enabled))
            self.model.tcp_enabled = bool(enabled)
            return True

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True

        def stop(self):
            calls.append(f"{self.name}-stop")
            self.active = False
            return True

        def start(self):
            calls.append(f"{self.name}-start")
            self.active = True
            return True

    widget = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(widget)
    lifecycle = _attach_resource_lifecycle_owner(widget)
    widget.trigger_controller = Trigger()
    widget.shortcut_mgr = Manager("shortcut")
    widget.hw_manager = Manager("hardware")
    widget.workflow_controller = SimpleNamespace(
        disconnect=lambda: calls.append("workflow-disconnect")
    )
    widget.export_controller = SimpleNamespace(
        disconnect=lambda: calls.append("export-disconnect")
    )
    widget.shutdown_coordinator = SimpleNamespace(
        disconnect=lambda: calls.append("shutdown-disconnect")
    )
    lifecycle._reusable_child_suspended = False
    lifecycle._lightweight_cleanup_done = False
    widget.configuration_view = SimpleNamespace(
        present_missing_configuration_prompt=lambda *_args, **_kwargs: True
    )
    widget.sequence_config = {}

    assert widget.close() is True
    assert calls == ["tcp-stop", "shortcut-stop", "hardware-stop"]
    assert widget.workflow_controller is not None
    assert widget.export_controller is not None

    widget.show()
    qapp.processEvents()
    assert calls[-3:] == ["shortcut-start", "hardware-start", ("tcp-resume", True)]
    assert widget.trigger_controller.is_active is True


def test_hidden_hardware_snapshot_does_not_start_default_disabled_manager():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False),
            stop_tcp=lambda: calls.append("tcp-stop") or True,
        ),
        shortcut_mgr=SimpleNamespace(
            active=False,
            stop=lambda: calls.append("shortcut-stop") or True,
            start=lambda: calls.append("shortcut-start") or True,
        ),
        hw_manager=SimpleNamespace(
            active=False,
            stop=lambda: calls.append("hardware-stop") or True,
            start=lambda: calls.append("hardware-start") or True,
        ),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_suspend_completed=set(),
        _reusable_resume_completed=set(),
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade)
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade)
    assert calls == []


def test_hidden_snapshot_uses_production_hardware_runtime_state(qapp):
    from base.unified_hid_device_manager import UnifiedHardwareManager
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []
    hardware = UnifiedHardwareManager()
    hardware.logger = SimpleNamespace(
        info=lambda *_args: None,
        warning=lambda *_args: None,
        error=lambda *_args: None,
    )
    def stop_hardware():
        calls.append("hardware-stop")
        hardware._scanner_enabled = False
        return True

    def start_hardware():
        calls.append("hardware-start")
        hardware._scanner_enabled = True
        return True

    hardware.stop = stop_hardware
    hardware.start = start_hardware
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=SimpleNamespace(active=False),
        hw_manager=hardware,
        toolsbar=SimpleNamespace(
            barcode_scanner_box=SimpleNamespace(isChecked=lambda: False)
        ),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_suspend_completed=set(),
        _reusable_resume_completed=set(),
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade)
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade)
    assert calls == []

    hardware._scanner_enabled = True
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade)
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade)
    assert calls == ["hardware-stop", "hardware-start"]


def test_hidden_suspend_retries_only_failed_stop_and_enabled_hardware_resumes_once():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []
    tcp_results = [False, True]
    model = SequenceTriggerModel(tcp_enabled=True)

    def stop_tcp():
        calls.append("tcp-stop")
        stopped = tcp_results.pop(0)
        if stopped:
            model.tcp_enabled = False
        return stopped

    def start_tcp(enabled, **_kwargs):
        calls.append(("tcp-start", enabled))
        model.tcp_enabled = bool(enabled)
        return True

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True

        def stop(self):
            calls.append(f"{self.name}-stop")
            self.active = False
            return True

        def start(self):
            calls.append(f"{self.name}-start")
            self.active = True
            return True

    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=model,
            stop_tcp=stop_tcp,
            set_tcp_enabled=start_tcp,
        ),
        shortcut_mgr=Manager("shortcut"),
        hw_manager=Manager("hardware"),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_suspend_completed=set(),
        _reusable_resume_completed=set(),
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert calls == [
        "tcp-stop",
        "shortcut-stop",
        "hardware-stop",
        "tcp-stop",
        "shortcut-start",
        "hardware-start",
        ("tcp-start", True),
    ]


@pytest.mark.parametrize(
    "failure", [False, RuntimeError("stop"), KeyboardInterrupt(), SystemExit(5)]
)
def test_partial_suspend_show_reverses_only_successfully_stopped_resources(failure):
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Resource:
        def __init__(self, name, stop_results):
            self.name = name
            self.active = True
            self.handle = object()
            self.stop_results = list(stop_results)

        def stop(self):
            calls.append((self.name, "stop"))
            result = self.stop_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            if result:
                self.active = False
                self.handle = None
            return result

        def start(self):
            calls.append((self.name, "start"))
            self.active = True
            self.handle = object()
            return True

    shortcut = Resource("shortcut", [failure])
    hardware = Resource("hardware", [True])

    class Trigger:
        def __init__(self):
            self.model = SimpleNamespace(
                tcp_enabled=True,
                tcp_running=True,
                tcp_server=object(),
                tcp_host="127.0.0.1",
                tcp_port=9001,
            )

        def stop_tcp(self):
            calls.append(("tcp", "stop"))
            self.model.tcp_enabled = False
            self.model.tcp_running = False
            self.model.tcp_server = None
            return True

        def set_tcp_enabled(self, enabled, **_kwargs):
            calls.append(("tcp", "start"))
            self.model.tcp_enabled = bool(enabled)
            self.model.tcp_running = bool(enabled)
            self.model.tcp_server = object() if enabled else None
            return True

    facade = SimpleNamespace(
        tcp_resource_port=Trigger(),
        shortcut_mgr=shortcut,
        hw_manager=hardware,
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert facade._reusable_resource_state == "SUSPENDING"
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert facade._reusable_resource_state == "ACTIVE"
    assert calls == [
        ("tcp", "stop"),
        ("shortcut", "stop"),
        ("hardware", "stop"),
        ("hardware", "start"),
        ("tcp", "start"),
    ]


@pytest.mark.parametrize(
    "start_failure",
    [False, RuntimeError("start"), KeyboardInterrupt(), SystemExit(6)],
)
def test_partial_resume_hide_stops_only_restored_resources_then_show_retries_all(
    start_failure,
):
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Resource:
        def __init__(self, name, start_results):
            self.name = name
            self.active = True
            self.handle = object()
            self.start_results = list(start_results)

        def stop(self):
            calls.append((self.name, "stop"))
            self.active = False
            self.handle = None
            return True

        def start(self):
            calls.append((self.name, "start"))
            result = self.start_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            if result:
                self.active = True
                self.handle = object()
            return result

    shortcut = Resource("shortcut", [start_failure, True])
    hardware = Resource("hardware", [True, True])

    class Trigger:
        def __init__(self):
            self.model = SimpleNamespace(
                tcp_enabled=True,
                tcp_running=True,
                tcp_server=object(),
                tcp_host="127.0.0.1",
                tcp_port=9001,
            )
            self.start_results = [True, True]

        def stop_tcp(self):
            calls.append(("tcp", "stop"))
            self.model.tcp_enabled = False
            self.model.tcp_running = False
            self.model.tcp_server = None
            return True

        def set_tcp_enabled(self, enabled, **_kwargs):
            calls.append(("tcp", "start"))
            result = self.start_results.pop(0)
            if result:
                self.model.tcp_enabled = bool(enabled)
                self.model.tcp_running = bool(enabled)
                self.model.tcp_server = object()
            return result

    facade = SimpleNamespace(
        tcp_resource_port=Trigger(),
        shortcut_mgr=shortcut,
        hw_manager=hardware,
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert facade._reusable_resource_state == "SUSPENDED"
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    assert facade._reusable_resource_state == "RESUMING"
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert facade._reusable_resource_state == "SUSPENDED"
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert facade._reusable_resource_state == "ACTIVE"
    assert calls == [
        ("tcp", "stop"),
        ("shortcut", "stop"),
        ("hardware", "stop"),
        ("shortcut", "start"),
        ("hardware", "start"),
        ("tcp", "start"),
        ("tcp", "stop"),
        ("hardware", "stop"),
        ("shortcut", "start"),
        ("hardware", "start"),
        ("tcp", "start"),
    ]


def test_suspend_rechecks_manager_identity_without_repeating_unchanged_success():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Resource:
        def __init__(self, name, stop_results):
            self.name = name
            self.active = True
            self.handle = object()
            self.stop_results = list(stop_results)

        def stop(self):
            calls.append((self.name, "stop"))
            result = self.stop_results.pop(0)
            if result:
                self.active = False
                self.handle = None
            return result

        def start(self):
            self.active = True
            self.handle = object()
            return True

    shortcut = Resource("shortcut-first", [True])
    hardware = Resource("hardware", [False, True])
    trigger = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False),
    )
    facade = SimpleNamespace(
        tcp_resource_port=trigger,
        shortcut_mgr=shortcut,
        hw_manager=hardware,
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    replacement = Resource("shortcut-replacement", [True])
    facade.shortcut_mgr = replacement
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert calls == [
        ("shortcut-first", "stop"),
        ("hardware", "stop"),
        ("shortcut-replacement", "stop"),
        ("hardware", "stop"),
    ]


def test_repeated_hide_rechecks_replaced_resource_after_fully_suspended():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Resource:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()

        def stop(self):
            calls.append((self.name, "stop"))
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.active = True
            self.handle = object()
            return True

    first_shortcut = Resource("shortcut-first")
    hardware = Resource("hardware")
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=first_shortcut,
        hw_manager=hardware,
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    facade.shortcut_mgr = Resource("shortcut-replacement")
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert calls == [
        ("shortcut-first", "stop"),
        ("hardware", "stop"),
        ("shortcut-replacement", "stop"),
    ]


def test_disabled_hidden_resource_replacement_is_stopped_but_not_resumed():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Manager:
        def __init__(self, name, active):
            self.name = name
            self.active = active
            self.handle = object() if active else None

        def stop(self):
            calls.append((self.name, "stop"))
            self.active = False
            self.handle = None
            return True

        def start(self):
            calls.append((self.name, "start"))
            self.active = True
            self.handle = object()
            return True

    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=Manager("disabled-original", False),
        hw_manager=Manager("hardware", False),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    replacement = Manager("active-replacement", True)
    facade.shortcut_mgr = replacement
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert replacement.active is False
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert calls == [("active-replacement", "stop")]


def test_reusable_suspend_journal_retries_failed_replaced_manager_instances():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            calls.append((self.name, "stop", self.stop_calls))
            if self.stop_calls == 1:
                return False
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.active = True
            self.handle = object()
            return True

    old = Manager("old")
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=old,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    replacement = Manager("replacement")
    facade.shortcut_mgr = replacement
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert old.stop_calls == 2
    assert replacement.stop_calls == 1
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert replacement.stop_calls == 2


def test_reusable_stop_and_start_verify_observable_postconditions():
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.start_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls > 1:
                self.active = False
                self.handle = None
            return True

        def start(self):
            self.start_calls += 1
            if self.start_calls > 1:
                self.active = True
                self.handle = object()
            return True

    shortcut = Manager()
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=shortcut,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True


def test_reusable_stop_reentrant_replacement_waits_for_next_hide_event():
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Manager:
        def __init__(self, on_stop=None):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

        def start(self):
            self.active = True
            self.handle = object()
            return True

    import gc
    import weakref

    replacement = Manager()
    old = Manager(lambda: setattr(holder["facade"], "shortcut_mgr", replacement))
    old_ref = weakref.ref(old)
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=old,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )
    holder["facade"] = facade

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert replacement.active is True
    assert replacement.stop_calls == 0
    del old
    gc.collect()
    assert old_ref() is None
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert replacement.active is False
    assert replacement.stop_calls == 1


def test_reusable_start_reentrant_replacement_rolls_back_exact_started_target():
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Replacement:
        def __init__(self):
            self.active = False
            self.handle = None
            self.start_calls = 0

        def stop(self):
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return True

    replacement = Replacement()

    class Old:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 2:
                return False
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.active = True
            self.handle = object()
            holder["facade"].shortcut_mgr = replacement
            return True

    old = Old()
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=old,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )
    holder["facade"] = facade

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    entry = facade._reusable_resource_journal["shortcut"]
    assert old in entry["pending_stops"].values()
    assert old.stop_calls == 2

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert old.stop_calls == 3
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert replacement.start_calls == 1


def test_reusable_resume_accepts_trusted_narrow_start_contract_without_state_port():
    from ui.sequence.sequence_widget import SequenceWindow

    class ContractManager:
        def __init__(self):
            self.start_calls = 0

        def stop(self):
            return True

        def start(self):
            self.start_calls += 1
            return True

    manager = ContractManager()
    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=manager,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=True,
        _reusable_resource_snapshot={
            "tcp_enabled": False,
            "tcp_host": None,
            "tcp_port": None,
        },
        _reusable_resource_state="SUSPENDED",
        _reusable_resource_journal={
            "shortcut": {
                "desired": True,
                "status": "STOPPED",
                "pending_stops": {},
            },
            "hardware": {
                "desired": False,
                "status": "STOPPED",
                "pending_stops": {},
            },
            "tcp": {
                "desired": False,
                "status": "STOPPED",
                "pending_stops": {},
            },
        },
        _reusable_trusted_running_ids={},
    )

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert manager.start_calls == 1
    assert facade._reusable_resource_state == "ACTIVE"


def test_reusable_controlled_setter_journals_every_stop_callback_identity():
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        def __init__(self, on_stop=None):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

        def start(self):
            self.active = True
            self.handle = object()
            return True

    intermediate = Manager()
    replacement = Manager()
    old = Manager(
        lambda: (
            setattr(holder["facade"], "shortcut_mgr", intermediate),
            setattr(holder["facade"], "shortcut_mgr", replacement),
        )
    )
    facade = Facade()
    facade._shortcut_mgr = old
    facade.hw_manager = SimpleNamespace(active=False, handle=None)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    holder["facade"] = facade

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert list(
        facade._reusable_resource_journal["shortcut"]["pending_stops"].values()
    ) == [intermediate, replacement]
    assert intermediate.stop_calls == replacement.stop_calls == 0

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert intermediate.stop_calls == replacement.stop_calls == 1


def test_reusable_start_and_rollback_callbacks_journal_every_identity_write():
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.start_calls = 0
            self.on_stop = None
            self.on_start = None

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            if self.on_start is not None:
                callback, self.on_start = self.on_start, None
                callback()
            return True

    start_intermediate = Manager()
    start_replacement = Manager()
    rollback_intermediate = Manager()
    rollback_replacement = Manager()
    original = Manager()

    def replace_during_start():
        facade = holder["facade"]
        facade.shortcut_mgr = start_intermediate
        facade.shortcut_mgr = start_replacement
        original.on_stop = lambda: (
            setattr(facade, "shortcut_mgr", rollback_intermediate),
            setattr(facade, "shortcut_mgr", rollback_replacement),
        )

    original.on_start = replace_during_start
    facade = Facade()
    facade._shortcut_mgr = original
    facade.hw_manager = SimpleNamespace(active=False, handle=None)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    holder["facade"] = facade

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    assert list(
        facade._reusable_resource_journal["shortcut"]["pending_stops"].values()
    ) == [
        start_intermediate,
        start_replacement,
        rollback_intermediate,
        rollback_replacement,
    ]

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert all(
        manager.stop_calls == 1
        for manager in (
            start_intermediate,
            start_replacement,
            rollback_intermediate,
            rollback_replacement,
        )
    )
    assert rollback_replacement.start_calls == 1


def test_reusable_final_review_catches_cross_resource_reinstall():
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

        @property
        def hw_manager(self):
            return self._hw_manager

        @hw_manager.setter
        def hw_manager(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "hardware", target
            )

        @property
        def tcp_resource_port(self):
            return self._tcp_resource_port

        @tcp_resource_port.setter
        def tcp_resource_port(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(self, "tcp", target)

    class Manager:
        def __init__(self, active=True, on_stop=None):
            self.active = active
            self.handle = object() if active else None
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    class Trigger:
        def __init__(self, active):
            self.model = SimpleNamespace(
                tcp_enabled=active,
                tcp_running=active,
                tcp_server=object() if active else None,
            )
            self._tcp_stop_journal = {}
            self.stop_calls = 0

        def stop_tcp(self):
            self.stop_calls += 1
            self.model.tcp_enabled = False
            self.model.tcp_running = False
            self.model.tcp_server = None
            return True

    replacement_shortcut = Manager()
    replacement_trigger = Trigger(True)

    def reinstall_earlier_resources():
        facade = holder["facade"]
        facade.shortcut_mgr = replacement_shortcut
        facade.tcp_resource_port = replacement_trigger

    facade = Facade()
    facade._shortcut_mgr = Manager(False)
    facade._hw_manager = Manager(on_stop=reinstall_earlier_resources)
    facade._tcp_resource_port = Trigger(False)
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    holder["facade"] = facade

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert replacement_shortcut.stop_calls == replacement_trigger.stop_calls == 0

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True
    assert replacement_shortcut.stop_calls == replacement_trigger.stop_calls == 1


def test_reusable_epoch_cas_rejects_final_review_gap_identity_write(monkeypatch):
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        def __init__(self, active):
            self.active = active
            self.handle = object() if active else None
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    replacement = Manager(True)
    facade = Facade()
    facade._shortcut_mgr = Manager(False)
    facade.hw_manager = Manager(False)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    holder["facade"] = facade
    original_review = SequenceResourceLifecycleController._review_reusable_resources_after_event

    def write_after_review(owner, journal, *, require_stopped):
        stable = original_review(
            owner, journal, require_stopped=require_stopped
        )
        owner.shortcut_mgr = replacement
        return stable

    monkeypatch.setattr(
        SequenceResourceLifecycleController,
        "_review_reusable_resources_after_event",
        staticmethod(write_after_review),
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert facade._reusable_resource_state == "SUSPENDING"
    assert replacement.stop_calls == 0


def test_reusable_suspended_late_admission_queues_one_cleanup_event():
    from ui.sequence.sequence_widget import SequenceWindow

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        def __init__(self, active):
            self.active = active
            self.handle = object() if active else None
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    facade = Facade()
    facade._shortcut_mgr = Manager(False)
    facade.hw_manager = Manager(False)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True

    replacement = Manager(True)
    facade.shortcut_mgr = replacement
    facade.shortcut_mgr = replacement
    assert facade._reusable_resource_state == "SUSPENDING"
    assert facade._reusable_cleanup_event_pending is True
    assert replacement.stop_calls == 0

    assert SequenceResourceLifecycleController._run_queued_reusable_cleanup(facade) is True
    assert facade._reusable_cleanup_event_pending is False
    assert replacement.stop_calls == 1
    assert facade._reusable_resource_state == "SUSPENDED"


def test_reusable_concurrent_setters_preserve_every_exact_pending_identity():
    from concurrent.futures import ThreadPoolExecutor
    from ui.sequence.sequence_widget import SequenceWindow

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        active = True
        handle = object()

        def stop(self):
            self.active = False
            self.handle = None
            return True

    facade = Facade()
    facade._shortcut_mgr = Manager()
    facade.hw_manager = SimpleNamespace(active=False, handle=None)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = True
    facade._reusable_resource_snapshot = {
        "shortcut": False,
        "hardware": False,
        "tcp_enabled": False,
    }
    facade._reusable_resource_state = "SUSPENDED"
    facade._reusable_resource_journal = {
        "shortcut": {
            "desired": False,
            "status": "STOPPED",
            "pending_stops": {},
        }
    }
    managers = [Manager() for _index in range(16)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda manager: setattr(facade, "shortcut_mgr", manager), managers))

    pending = facade._reusable_resource_journal["shortcut"]["pending_stops"]
    assert all(manager in pending.values() for manager in managers)
    assert facade._reusable_resource_state == "SUSPENDING"
    assert facade._reusable_cleanup_event_pending is True


def test_sequence_window_tcp_mirror_has_one_canonical_class_identity(monkeypatch):
    from ui.sequence.sequence_widget import SequenceWindow

    class Server:
        pass

    server = Server()
    server_ref = weakref.ref(server)
    monkeypatch.setattr(SequenceWindow, "tcp_server", None)

    SequenceWindow.tcp_server = server
    assert SequenceWindow.tcp_server is server

    SequenceWindow.tcp_server = None
    assert SequenceWindow.tcp_server is None
    del server
    gc.collect()
    assert server_ref() is None


def test_reusable_late_admission_pressure_releases_every_completed_identity():
    from ui.sequence.sequence_widget import SequenceWindow

    class Facade:
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    class Manager:
        def __init__(self, active=False):
            self.active = active
            self.handle = object() if active else None

        def stop(self):
            self.active = False
            self.handle = None
            return True

    facade = Facade()
    facade._shortcut_mgr = Manager(False)
    facade.hw_manager = Manager(False)
    facade.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    facade._reusable_child_suspended = False
    facade._reusable_resource_snapshot = None
    facade._reusable_resource_state = "ACTIVE"
    facade._reusable_resource_journal = {}
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True

    previous = facade.shortcut_mgr
    retired = []
    for _index in range(32):
        replacement = Manager(True)
        retired.append(weakref.ref(previous))
        facade.shortcut_mgr = replacement
        assert facade._reusable_cleanup_event_pending is True
        assert SequenceResourceLifecycleController._run_queued_reusable_cleanup(facade) is True
        assert facade._reusable_resource_state == "SUSPENDED"
        assert facade._reusable_resource_journal["shortcut"][
            "pending_stops"
        ] == {}
        previous = replacement
        del replacement
        gc.collect()
        assert all(reference() is None for reference in retired)

    previous_ref = weakref.ref(previous)
    facade.shortcut_mgr = None
    assert SequenceResourceLifecycleController._run_queued_reusable_cleanup(facade) is True
    del previous
    gc.collect()
    assert previous_ref() is None


def _loop5_reusable_owner(manager):
    from ui.sequence.sequence_widget import SequenceWindow

    class Owner(QObject):
        @property
        def shortcut_mgr(self):
            return self._shortcut_mgr

        @shortcut_mgr.setter
        def shortcut_mgr(self, target):
            SequenceResourceLifecycleController._set_reusable_resource_identity(
                self, "shortcut", target
            )

    owner = Owner()
    owner._shortcut_mgr = manager
    owner.hw_manager = SimpleNamespace(active=False, handle=None, stop=lambda: True)
    owner.tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    owner._reusable_resource_lock = RLock()
    owner._reusable_resource_epoch = 0
    owner._reusable_child_suspended = True
    owner._reusable_resource_snapshot = {
        "shortcut": False,
        "hardware": False,
        "tcp_enabled": False,
        "tcp_host": None,
        "tcp_port": None,
    }
    owner._reusable_resource_state = "SUSPENDED"
    owner._reusable_resource_journal = {
        "shortcut": {
            "desired": False,
            "status": "STOPPED",
            "pending_stops": {},
        },
        "hardware": {
            "desired": False,
            "status": "STOPPED",
            "pending_stops": {},
        },
        "tcp": {
            "desired": False,
            "status": "STOPPED",
            "pending_stops": {},
        },
    }
    owner._reusable_trusted_running_ids = {}
    owner._reusable_cleanup_event_pending = False
    owner._reusable_cleanup_retry_limit = 3
    owner._reusable_cleanup_retry_delays_ms = (0, 5, 10)
    SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
    return owner


@pytest.mark.parametrize(
    "first_failure",
    [False, RuntimeError("stop"), KeyboardInterrupt(), SystemExit(41)],
)
def test_reusable_queued_cleanup_baseexception_retries_and_converges(
    qapp, first_failure
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.results = [first_failure, True]
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            if result:
                self.active = False
                self.handle = None
            return result

    owner = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    manager = Manager()
    owner.shortcut_mgr = manager

    QTest.qWait(80)
    assert manager.stop_calls == 2
    assert owner._reusable_resource_state == "SUSPENDED"
    assert owner._reusable_cleanup_event_pending is False
    assert owner._reusable_resource_journal["shortcut"]["pending_stops"] == {}


def test_reusable_persistent_failure_stops_at_bound_and_explicit_sync_starts_new_round(
    qapp,
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.results = [False, False, False]
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            result = self.results.pop(0)
            if result:
                self.active = False
                self.handle = None
            return result

    owner = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    manager = Manager()
    owner.shortcut_mgr = manager

    QTest.qWait(100)
    assert manager.stop_calls == owner._reusable_cleanup_retry_limit == 3
    assert owner._reusable_cleanup_event_pending is False
    assert manager in owner._reusable_resource_journal["shortcut"][
        "pending_stops"
    ].values()

    manager.results = [True]
    assert SequenceResourceLifecycleController._synchronize_reusable_cleanup(owner) is True
    QTest.qWait(30)
    assert manager.stop_calls == 4
    assert owner._reusable_resource_state == "SUSPENDED"
    assert owner._reusable_resource_journal["shortcut"]["pending_stops"] == {}


def test_reusable_queued_cleanup_epoch_conflict_requeues_one_bounded_round(
    qapp, monkeypatch
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    owner = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    first = Manager()
    replacement = Manager()
    original_review = SequenceResourceLifecycleController._review_reusable_resources_after_event
    conflicted = []

    def conflict_once(actual_owner, journal, *, require_stopped):
        stable = original_review(
            actual_owner, journal, require_stopped=require_stopped
        )
        if not conflicted:
            conflicted.append(True)
            actual_owner.shortcut_mgr = replacement
        return stable

    monkeypatch.setattr(
        SequenceResourceLifecycleController,
        "_review_reusable_resources_after_event",
        staticmethod(conflict_once),
    )
    owner.shortcut_mgr = first

    QTest.qWait(100)
    assert first.stop_calls == 1
    assert replacement.stop_calls == 1
    assert owner._reusable_resource_state == "SUSPENDED"
    assert owner._reusable_cleanup_event_pending is False


def test_reusable_cleanup_event_owner_gc_and_cancel_release_pending_identity(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        active = True
        handle = object()

        def stop(self):
            return False

    owner = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    owner._reusable_cleanup_retry_delays_ms = (100, 100, 100)
    manager = Manager()
    owner.shortcut_mgr = manager
    owner_ref = weakref.ref(owner)
    manager_ref = weakref.ref(manager)
    owner.shortcut_mgr = None
    SequenceResourceLifecycleController._cancel_reusable_cleanup_events(
        owner, release_pending=True
    )
    del manager
    del owner
    gc.collect()

    assert owner_ref() is None
    assert manager_ref() is None
    QTest.qWait(120)


def test_reusable_external_hide_failure_starts_a_fresh_bounded_cleanup_round(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.results = [False, True]
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            result = self.results.pop(0)
            if result:
                self.active = False
                self.handle = None
            return result

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_child_suspended = False
    owner._reusable_resource_snapshot = None
    owner._reusable_resource_state = "ACTIVE"
    owner._reusable_resource_journal = {}

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(owner) is False
    assert owner._reusable_cleanup_event_pending is True

    QTest.qWait(30)
    assert manager.stop_calls == 2
    assert owner._reusable_resource_state == "SUSPENDED"
    assert owner._reusable_cleanup_event_pending is False


def test_reusable_external_show_partial_failure_starts_cleanup_round(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return False

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_journal["shortcut"]["desired"] = True
    owner._reusable_resource_snapshot["shortcut"] = True

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(owner) is False
    assert manager.active is True
    assert owner._reusable_cleanup_event_pending is True

    QTest.qWait(30)
    assert manager.start_calls == 1
    assert manager.stop_calls == 1
    assert owner._reusable_resource_state == "SUSPENDED"
    assert owner._reusable_cleanup_event_pending is False


def test_reusable_show_cancels_queued_hide_generation_before_event_delivery(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.start_calls = 0
            self.stop_calls = 0

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    owner = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True
    manager = Manager()
    owner.shortcut_mgr = manager
    hide_generation = owner._reusable_cleanup_generation
    assert owner._reusable_cleanup_event_pending is True

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(owner) is True
    resume_generation = owner._reusable_cleanup_generation
    assert resume_generation > hide_generation
    assert owner._reusable_cleanup_event_pending is False
    assert owner._reusable_resource_state == "ACTIVE"

    stop_calls_after_resume = manager.stop_calls
    QTest.qWait(30)
    assert manager.stop_calls == stop_calls_after_resume
    assert manager.active is True
    assert owner._reusable_resource_state == "ACTIVE"


def test_reusable_partial_show_replaces_old_hide_retry_with_fresh_generation(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        active = False
        handle = None

        def __init__(self):
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            return False

        def stop(self):
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True
    old_generation = SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(owner)

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(owner) is False
    assert owner._reusable_cleanup_generation > old_generation
    assert owner._reusable_cleanup_event_pending is True


def test_reusable_queued_cleanup_ignores_retained_native_deleted_owner(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        active = True
        handle = object()

        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            return False

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    with owner._reusable_resource_lock:
        generation = SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(owner)
    SequenceResourceLifecycleController._queue_reusable_cleanup_event(
        owner, generation, delay_ms=20
    )
    sip.delete(owner)
    assert sip.isdeleted(owner)

    QTest.qWait(40)
    assert manager.stop_calls == 0


def test_reusable_pending_capacity_rejects_before_backing_identity_mutation():
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        active = True
        handle = object()

        def stop(self):
            return False

    owner = _loop5_reusable_owner(Manager())
    owner._reusable_pending_identity_limit = 3
    accepted = [owner.shortcut_mgr]
    rejected = []

    for _index in range(100):
        candidate = Manager()
        previous = owner.shortcut_mgr
        owner.shortcut_mgr = candidate
        if owner.shortcut_mgr is candidate:
            accepted.append(candidate)
        else:
            rejected.append(candidate)
            assert owner.shortcut_mgr is previous
        pending = owner._reusable_resource_journal["shortcut"]["pending_stops"]
        assert len(pending) <= 3

    assert rejected
    assert len(accepted) == 3

    for manager in accepted:
        manager.active = False
        manager.handle = None
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(owner) is True
    assert owner._reusable_resource_journal["shortcut"]["pending_stops"] == {}

    after_release = Manager()
    owner.shortcut_mgr = after_release
    assert owner.shortcut_mgr is after_release
    released_pending = owner._reusable_resource_journal["shortcut"][
        "pending_stops"
    ]
    assert after_release in released_pending.values()
    assert len(released_pending) <= owner._reusable_pending_identity_limit


def test_reusable_cleanup_and_setter_share_capacity_commit_lock():
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self, *, blocking=False):
            self.active = True
            self.handle = object()
            self.blocking = blocking

        def stop(self):
            if self.blocking:
                entered.set()
                assert release.wait(2)
            self.active = False
            self.handle = None
            return True

    original = Manager(blocking=True)
    owner = _loop5_reusable_owner(original)
    owner._reusable_pending_identity_limit = 1
    owner._reusable_resource_journal["shortcut"]["pending_stops"] = {
        id(original): original
    }
    replacement = Manager()

    with ThreadPoolExecutor(max_workers=1) as executor:
        cleanup = executor.submit(
            SequenceResourceLifecycleController._suspend_reusable_child_resources, owner
        )
        assert entered.wait(2)
        owner.shortcut_mgr = replacement
        assert owner.shortcut_mgr is original
        assert len(
            owner._reusable_resource_journal["shortcut"]["pending_stops"]
        ) == 1
        release.set()
        assert cleanup.result(timeout=2) is True

    assert owner._reusable_resource_journal["shortcut"]["pending_stops"] == {}
    owner.shortcut_mgr = replacement
    assert owner.shortcut_mgr is replacement
    assert len(
        owner._reusable_resource_journal["shortcut"]["pending_stops"]
    ) <= 1


def test_show_during_inflight_suspend_returns_pending_and_resumes_once(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.start_calls = 0

        def stop(self):
            self.stop_calls += 1
            entered.set()
            assert release.wait(2)
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_child_suspended = False
    owner._reusable_resource_snapshot = None
    owner._reusable_resource_state = "ACTIVE"
    owner._reusable_resource_journal = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        suspending = executor.submit(
            SequenceResourceLifecycleController._suspend_reusable_child_resources, owner
        )
        assert entered.wait(2)
        delayed_release = Timer(0.3, release.set)
        delayed_release.start()
        started = __import__("time").monotonic()
        assert SequenceResourceLifecycleController._resume_reusable_child_resources(owner) is False
        elapsed = __import__("time").monotonic() - started
        delayed_release.join(1)
        assert elapsed < 0.15
        assert owner._reusable_resume_pending is True
        assert manager.start_calls == 0
        assert suspending.result(timeout=2) is False

    QTest.qWait(80)
    assert manager.stop_calls == 1
    assert manager.start_calls == 1
    assert manager.active is True
    assert owner._reusable_resource_state == "ACTIVE"
    assert owner._reusable_resume_pending is False


def test_hide_during_inflight_resume_returns_pending_and_suspends_once(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        assert SequenceResourceLifecycleController._suspend_reusable_child_resources(owner) is False
        assert SequenceResourceLifecycleController._suspend_reusable_child_resources(owner) is False
        release.set()
        assert resuming.result(timeout=2) is False

    QTest.qWait(80)
    assert manager.start_calls == 1
    assert manager.stop_calls == 1
    assert manager.active is False
    assert owner._reusable_resource_state == "SUSPENDED"


def test_real_show_event_during_inflight_suspend_is_nonblocking_pending(qapp):
    import time
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.start_calls = 0

        def stop(self):
            entered.set()
            assert release.wait(2)
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return True

    widget = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(widget)
    lifecycle = _attach_resource_lifecycle_owner(widget)
    manager = Manager()
    lifecycle._shortcut_mgr = manager
    lifecycle._hw_manager = SimpleNamespace(active=False, handle=None, stop=lambda: True)
    lifecycle._tcp_resource_port = SimpleNamespace(
        model=SequenceTriggerModel(tcp_enabled=False)
    )
    lifecycle._reusable_child_suspended = False
    lifecycle._reusable_resource_snapshot = None
    lifecycle._reusable_resource_state = "ACTIVE"
    lifecycle._reusable_resource_journal = {}
    widget.configuration_view = SimpleNamespace(
        present_missing_configuration_prompt=lambda *_args, **_kwargs: True
    )
    widget.sequence_config = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        suspending = executor.submit(
            lifecycle._suspend_reusable_child_resources
        )
        assert entered.wait(2)
        delayed_release = Timer(0.3, release.set)
        delayed_release.start()
        started = time.monotonic()
        SequenceWindow.showEvent(widget, QShowEvent())
        elapsed = time.monotonic() - started
        delayed_release.join(1)
        assert elapsed < 0.15
        assert lifecycle._reusable_resume_pending is True
        assert suspending.result(timeout=2) is False

    QTest.qWait(80)
    assert manager.start_calls == 1
    assert lifecycle._reusable_resource_state == "ACTIVE"


def test_destroy_during_inflight_suspend_cancels_late_compensation(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        active = True
        handle = object()

        def __init__(self):
            self.start_calls = 0

        def stop(self):
            entered.set()
            assert release.wait(2)
            self.active = False
            self.handle = None
            return True

        def start(self):
            self.start_calls += 1
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_child_suspended = False
    owner._reusable_resource_snapshot = None
    owner._reusable_resource_state = "ACTIVE"
    owner._reusable_resource_journal = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        suspending = executor.submit(
            SequenceResourceLifecycleController._suspend_reusable_child_resources, owner
        )
        assert entered.wait(2)
        delayed_release = Timer(0.3, release.set)
        delayed_release.start()
        assert SequenceResourceLifecycleController._resume_reusable_child_resources(owner) is False
        delayed_release.join(1)
        sip.delete(owner)
        assert suspending.result(timeout=2) is False

    QTest.qWait(50)
    assert manager.start_calls == 0


def test_reusable_queue_bridge_is_instance_owned_across_windows(qapp):
    import gc
    from weakref import ref
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    first = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    second = _loop5_reusable_owner(SimpleNamespace(active=False, handle=None))
    SequenceResourceLifecycleController._reusable_resource_identity_lock(first)
    SequenceResourceLifecycleController._reusable_resource_identity_lock(second)

    assert not hasattr(
        sequence_widget_module, "_REUSABLE_CLEANUP_DISPATCHER"
    )
    assert first._reusable_cleanup_dispatcher is not (
        second._reusable_cleanup_dispatcher
    )

    first_ref = ref(first)
    with first._reusable_resource_lock:
        generation = SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(first)
    SequenceResourceLifecycleController._queue_reusable_cleanup_event(
        first, generation, delay_ms=50
    )
    del first
    gc.collect()
    assert first_ref() is None
    QTest.qWait(70)


@pytest.mark.parametrize(
    "stop_outcome",
    [
        True,
        False,
        RuntimeError("stop"),
        KeyboardInterrupt(),
        SystemExit(79),
    ],
)
def test_native_destroy_during_inflight_start_rolls_back_exact_target(
    qapp, stop_outcome
):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0
            self.stop_outcome = stop_outcome

        def start(self):
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            outcome = self.stop_outcome
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                self.active = False
                self.handle = None
            return outcome

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        sip.delete(owner)
        release.set()
        assert resuming.result(timeout=2) is False

    assert manager.stop_calls == 1
    detached = getattr(owner, "_reusable_detached_pending_stops", {})
    if stop_outcome is True:
        assert manager.active is False
        assert detached == {}
    else:
        exact_key = ("shortcut", id(manager))
        assert detached.get(exact_key) is manager
        manager.stop_outcome = True
        assert SequenceResourceLifecycleController._retry_detached_reusable_stops(owner) is False
        QTest.qWait(50)
        assert manager.active is False
        assert owner._reusable_detached_pending_stops == {}


def test_native_destroy_inflight_start_successful_rollback_releases_on_gc(qapp):
    import gc
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0

        def start(self):
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    manager_ref = weakref.ref(manager)
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        sip.delete(owner)
        release.set()
        assert resuming.result(timeout=2) is False

    assert manager.stop_calls == 1
    del manager
    del owner
    gc.collect()
    assert manager_ref() is None


def test_native_destroy_failed_rollback_retries_after_owner_wrapper_gc(qapp):
    import gc
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0

        def start(self):
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return False
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner_ref = weakref.ref(owner)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        sip.delete(owner)
        release.set()
        assert resuming.result(timeout=2) is False

    del owner
    gc.collect()
    assert owner_ref() is None

    QTest.qWait(50)
    assert manager.stop_calls == 2
    assert manager.active is False
    assert not hasattr(manager, "_sequence_detached_cleanup_token")


def test_detached_reservation_shares_resource_capacity_with_regular_journal():
    from ui.sequence.sequence_widget import SequenceWindow

    detached = object()
    regular = object()
    owner = _loop5_reusable_owner(detached)
    owner._reusable_pending_identity_limit = 1
    entry = owner._reusable_resource_journal["shortcut"]

    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", detached
    ) is True
    assert SequenceResourceLifecycleController._admit_reusable_pending_locked(
        owner, entry, regular
    ) is False
    assert entry["pending_stops"] == {}

    SequenceResourceLifecycleController._release_detached_reusable_target(
        owner, "shortcut", detached
    )
    assert SequenceResourceLifecycleController._admit_reusable_pending_locked(
        owner, entry, regular
    ) is True
    assert entry["pending_stops"] == {id(regular): regular}


@pytest.mark.parametrize("eventual_success", [True, False])
def test_native_destroy_detached_token_retries_with_a_strict_bound(
    qapp, eventual_success
):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0
            self.allow_stop = eventual_success

        def start(self):
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            stopped = self.allow_stop and self.stop_calls >= 2
            if stopped:
                self.active = False
                self.handle = None
            return stopped

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        sip.delete(owner)
        release.set()
        assert resuming.result(timeout=2) is False

    QTest.qWait(50)
    if eventual_success:
        assert manager.stop_calls == 2
        assert manager.active is False
        assert owner._reusable_detached_pending_stops == {}
    else:
        assert manager.stop_calls == 1 + owner._reusable_cleanup_retry_limit
        exact_key = ("shortcut", id(manager))
        assert owner._reusable_detached_pending_stops.get(exact_key) is manager
        assert owner._reusable_detached_cleanup_tokens.get(exact_key) is (
            manager._sequence_detached_cleanup_token
        )
        manager.allow_stop = True
        assert SequenceResourceLifecycleController._retry_detached_reusable_stops(owner) is False
        QTest.qWait(50)
        assert owner._reusable_detached_pending_stops == {}


def test_app_broker_capacity_is_reserved_before_start_across_windows(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class BlockingManager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.active = False
            self.handle = None
            return True

    class OtherManager(BlockingManager):
        def start(self):
            self.start_calls += 1
            self.active = True
            self.handle = object()
            return True

    first_manager = BlockingManager()
    second_manager = OtherManager()
    first = _loop5_reusable_owner(first_manager)
    second = _loop5_reusable_owner(second_manager)
    for owner in (first, second):
        owner._reusable_resource_snapshot["shortcut"] = True
        owner._reusable_resource_journal["shortcut"]["desired"] = True
    broker = SequenceResourceLifecycleController._reusable_detached_broker(first)
    original_capacity = broker.capacity
    broker.capacity = 1
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            resuming = executor.submit(
                SequenceResourceLifecycleController._resume_reusable_child_resources, first
            )
            assert entered.wait(2)
            assert SequenceResourceLifecycleController._resume_reusable_child_resources(second) is False
            assert second_manager.start_calls == 0
            assert broker.pending_count == 1
            release.set()
            assert resuming.result(timeout=2) is True
        assert broker.pending_count == 0
    finally:
        release.set()
        broker.capacity = original_capacity


def test_app_broker_owns_detached_target_after_owner_and_external_refs_gc(qapp):
    import gc
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()

    class Manager:
        def __init__(self):
            self.active = False
            self.handle = None
            self.stop_calls = 0
            self.allow_stop = False

        def start(self):
            entered.set()
            assert release.wait(2)
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            self.stop_calls += 1
            if not self.allow_stop:
                return False
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_resource_snapshot["shortcut"] = True
    owner._reusable_resource_journal["shortcut"]["desired"] = True
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    owner_ref = weakref.ref(owner)
    manager_ref = weakref.ref(manager)

    with ThreadPoolExecutor(max_workers=1) as executor:
        resuming = executor.submit(
            SequenceResourceLifecycleController._resume_reusable_child_resources, owner
        )
        assert entered.wait(2)
        sip.delete(owner)
        release.set()
        assert resuming.result(timeout=2) is False

    QTest.qWait(50)
    assert broker.pending_count == 1
    del owner
    del manager
    gc.collect()
    assert owner_ref() is None
    assert manager_ref() is not None

    retained = manager_ref()
    retained.allow_stop = True
    del retained
    assert broker.retry_pending() == 1
    QTest.qWait(50)
    gc.collect()
    assert broker.pending_count == 0
    assert manager_ref() is None


def test_detached_token_is_activated_before_broker_promotion_and_keeps_provisional_demand(
    qapp, monkeypatch
):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    observed_entry_types = []
    original_activate = (
        sequence_widget_module._DetachedReusableCleanupToken.activate
    )

    def activate_after_provisional_retry(token, target_thread):
        observed_entry_types.append(type(broker._registry[exact_key]).__name__)
        assert broker.retry_pending() == 1
        return original_activate(token, target_thread)

    monkeypatch.setattr(
        sequence_widget_module._DetachedReusableCleanupToken,
        "activate",
        activate_after_provisional_retry,
    )
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    try:
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, "shortcut", manager
        )
        QTest.qWait(50)

        assert observed_entry_types == ["_ReusableDetachedProvisional"]
        assert manager.stop_calls == 1
        assert broker.pending_count == baseline
        assert owner._reusable_detached_pending_stops == {}
    finally:
        SequenceResourceLifecycleController._release_detached_reusable_target(
            owner, "shortcut", manager
        )


@pytest.mark.parametrize(
    "activation_error",
    [RuntimeError("activate"), KeyboardInterrupt(), SystemExit(87)],
)
def test_detached_activation_baseexception_keeps_provisional_retryable(
    qapp, monkeypatch, activation_error
):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    original_activate = (
        sequence_widget_module._DetachedReusableCleanupToken.activate
    )
    activation_calls = 0

    def fail_once(token, target_thread):
        nonlocal activation_calls
        activation_calls += 1
        if activation_calls == 1:
            raise activation_error
        return original_activate(token, target_thread)

    monkeypatch.setattr(
        sequence_widget_module._DetachedReusableCleanupToken,
        "activate",
        fail_once,
    )
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    try:
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, "shortcut", manager
        )
        assert broker.pending_count == baseline + 1
        assert manager.stop_calls == 0

        assert SequenceResourceLifecycleController._retry_detached_reusable_stops(owner) is False
        QTest.qWait(50)

        assert activation_calls == 2
        assert manager.stop_calls == 1
        assert broker.pending_count == baseline
        assert owner._reusable_detached_pending_stops == {}
    finally:
        SequenceResourceLifecycleController._release_detached_reusable_target(
            owner, "shortcut", manager
        )


@pytest.mark.parametrize(
    "factory_error",
    [RuntimeError("factory"), KeyboardInterrupt(), SystemExit(89)],
)
def test_detached_token_factory_baseexception_keeps_provisional_retryable(
    qapp, monkeypatch, factory_error
):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    original_token_type = (
        sequence_widget_module._DetachedReusableCleanupToken
    )

    class FailOnceToken(original_token_type):
        construction_calls = 0

        def __new__(cls, *args, **kwargs):
            cls.construction_calls += 1
            if cls.construction_calls == 1:
                raise factory_error
            return super().__new__(cls)

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    monkeypatch.setattr(
        sequence_widget_module,
        "_DetachedReusableCleanupToken",
        FailOnceToken,
    )
    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    try:
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, "shortcut", manager
        )
        assert broker.pending_count == baseline + 1
        assert manager.stop_calls == 0

        assert SequenceResourceLifecycleController._retry_detached_reusable_stops(owner) is False
        QTest.qWait(50)

        assert FailOnceToken.construction_calls == 2
        assert manager.stop_calls == 1
        assert broker.pending_count == baseline
    finally:
        SequenceResourceLifecycleController._release_detached_reusable_target(
            owner, "shortcut", manager
        )


def test_detached_native_deleted_during_activation_keeps_provisional_retryable(
    qapp, monkeypatch
):
    from PyQt5 import sip
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    original_activate = (
        sequence_widget_module._DetachedReusableCleanupToken.activate
    )
    activation_calls = 0

    def native_delete_once(token, target_thread):
        nonlocal activation_calls
        activation_calls += 1
        if activation_calls == 1:
            sip.delete(token)
            raise RuntimeError("native token deleted during activation")
        return original_activate(token, target_thread)

    monkeypatch.setattr(
        sequence_widget_module._DetachedReusableCleanupToken,
        "activate",
        native_delete_once,
    )
    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    try:
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, "shortcut", manager
        )
        assert broker.pending_count == baseline + 1
        assert SequenceResourceLifecycleController._retry_detached_reusable_stops(owner) is False
        QTest.qWait(50)

        assert activation_calls == 2
        assert manager.stop_calls == 1
        assert broker.pending_count == baseline
    finally:
        SequenceResourceLifecycleController._release_detached_reusable_target(
            owner, "shortcut", manager
        )


@pytest.mark.parametrize(
    "emit_error",
    [RuntimeError("emit"), KeyboardInterrupt(), SystemExit(90)],
)
def test_detached_requested_emit_baseexception_is_contained_and_retryable(
    qapp, monkeypatch, emit_error
):
    from ui.sequence.sequence_widget import SequenceWindow

    class SignalProbe(QObject):
        requested = pyqtSignal(int)

    signal_type = type(SignalProbe().requested)
    original_emit = signal_type.emit
    failures = [emit_error]

    def fail_requested_once(bound_signal, *args):
        if bound_signal.signal == "2requested(int)" and failures:
            raise failures.pop(0)
        return original_emit(bound_signal, *args)

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    monkeypatch.setattr(signal_type, "emit", fail_requested_once)
    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    try:
        SequenceResourceLifecycleController._schedule_detached_reusable_retry(
            owner, "shortcut", manager
        )
        QTest.qWait(20)

        assert failures == []
        assert manager.stop_calls == 0
        token = owner._reusable_detached_cleanup_tokens[
            ("shortcut", id(manager))
        ]
        assert token._state == token.IDLE
        assert broker.retry_pending() == 1
        QTest.qWait(50)

        assert manager.stop_calls == 1
        assert broker.pending_count == baseline
    finally:
        SequenceResourceLifecycleController._release_detached_reusable_target(
            owner, "shortcut", manager
        )


@pytest.mark.parametrize("delivery", ["retry", "teardown"])
def test_broker_rebuilds_provisional_after_factory_failure_and_owner_gc(
    qapp, monkeypatch, delivery
):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    original_token_type = (
        sequence_widget_module._DetachedReusableCleanupToken
    )

    class FailOnceToken(original_token_type):
        construction_calls = 0

        def __new__(cls, *args, **kwargs):
            cls.construction_calls += 1
            if cls.construction_calls == 1:
                raise SystemExit(101)
            return super().__new__(cls)

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    monkeypatch.setattr(
        sequence_widget_module,
        "_DetachedReusableCleanupToken",
        FailOnceToken,
    )
    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner_ref = weakref.ref(owner)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    assert FailOnceToken.construction_calls == 1
    assert broker.pending_count == baseline + 1

    sip.delete(owner)
    del owner
    gc.collect()
    assert owner_ref() is None
    try:
        if delivery == "retry":
            assert broker.retry_pending() == 1
            QTest.qWait(50)
        else:
            broker._on_app_teardown()

        assert FailOnceToken.construction_calls == 2
        assert manager.stop_calls == 1
        assert manager.active is False
        assert broker.pending_count == baseline
    finally:
        broker.release_provisional(exact_key, manager)


def test_broker_demotes_and_rebuilds_native_deleted_idle_token(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False

        def stop(self):
            self.stop_calls += 1
            if not self.allow_stop:
                return False
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(30)
    old_token = owner._reusable_detached_cleanup_tokens[exact_key]
    assert old_token._state == old_token.IDLE
    sip.delete(old_token)
    manager.allow_stop = True

    assert broker.retry_pending() == 1
    QTest.qWait(50)

    assert manager.stop_calls == 2
    assert broker.pending_count == baseline
    assert owner._reusable_detached_pending_stops == {}
    assert owner._reusable_detached_cleanup_tokens == {}
    assert not hasattr(manager, "_sequence_detached_cleanup_token")


def test_broker_demotes_and_rebuilds_native_deleted_scheduled_token(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    old_token = owner._reusable_detached_cleanup_tokens[exact_key]
    assert old_token._state == old_token.SCHEDULED
    sip.delete(old_token)

    assert broker.retry_pending() == 1
    QTest.qWait(50)

    assert manager.stop_calls == 1
    assert manager.active is False
    assert broker.pending_count == baseline
    assert owner._reusable_detached_cleanup_tokens == {}


def test_broker_rebuilds_native_deleted_running_token_without_concurrency(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.concurrent = 0
            self.max_concurrent = 0

        def stop(self):
            self.stop_calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            try:
                if self.stop_calls == 1:
                    old_token = holder["owner"]._reusable_detached_cleanup_tokens[
                        holder["exact_key"]
                    ]
                    assert old_token._state == old_token.RUNNING
                    sip.delete(old_token)
                    assert holder["broker"].retry_pending() == 1
                    return False
                self.active = False
                self.handle = None
                return True
            finally:
                self.concurrent -= 1

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    holder.update(owner=owner, broker=broker, exact_key=exact_key)
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(75)

    assert manager.stop_calls == 2
    assert manager.max_concurrent == 1
    assert broker.pending_count == baseline
    assert owner._reusable_detached_cleanup_tokens == {}


def test_concurrent_worker_retries_rebuild_one_native_deleted_token(qapp, monkeypatch):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    original_token_type = (
        sequence_widget_module._DetachedReusableCleanupToken
    )

    class CountingToken(original_token_type):
        construction_calls = 0

        def __init__(self, *args, **kwargs):
            type(self).construction_calls += 1
            super().__init__(*args, **kwargs)

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False

        def stop(self):
            self.stop_calls += 1
            if not self.allow_stop:
                return False
            self.active = False
            self.handle = None
            return True

    monkeypatch.setattr(
        sequence_widget_module,
        "_DetachedReusableCleanupToken",
        CountingToken,
    )
    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(30)
    assert CountingToken.construction_calls == 1
    sip.delete(owner._reusable_detached_cleanup_tokens[exact_key])
    manager.allow_stop = True

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: broker.retry_pending(), range(24)))

    assert manager.stop_calls == 1
    assert all(result == 1 for result in results)
    QTest.qWait(75)
    assert CountingToken.construction_calls == 2
    assert manager.stop_calls == 2
    assert broker.pending_count == baseline


@pytest.mark.parametrize(
    "teardown_outcome",
    [True, KeyboardInterrupt(), SystemExit(103)],
)
def test_worker_teardown_of_idle_token_runs_only_on_broker_affinity(
    qapp, teardown_outcome
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.outcome = False

        def stop(self):
            self.stop_calls += 1
            outcome = self.outcome
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                self.active = False
                self.handle = None
            return outcome

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(30)
    assert manager.stop_calls == 1
    manager.outcome = teardown_outcome

    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_in_worker = executor.submit(
            lambda: broker._on_app_teardown() or manager.stop_calls
        ).result(timeout=2)

    assert observed_in_worker == 1
    assert manager.stop_calls == 1
    QTest.qWait(50)
    assert manager.stop_calls == 2
    if teardown_outcome is True:
        assert broker.pending_count == baseline
        return

    assert broker.pending_count == baseline + 1
    manager.outcome = True
    assert broker.retry_pending() == 1
    QTest.qWait(50)
    assert manager.stop_calls == 3
    assert broker.pending_count == baseline


def test_worker_teardown_of_scheduled_token_is_queued_to_broker_affinity(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_in_worker = executor.submit(
            lambda: broker._on_app_teardown() or manager.stop_calls
        ).result(timeout=2)

    assert observed_in_worker == 0
    assert manager.stop_calls == 0
    QTest.qWait(50)
    assert manager.stop_calls == 1
    assert broker.pending_count == baseline


def test_worker_teardown_during_running_token_never_stops_concurrently(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.concurrent = 0
            self.max_concurrent = 0
            self.worker_observations = []

        def stop(self):
            self.stop_calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            try:
                if self.stop_calls == 1:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        observed = executor.submit(
                            lambda: holder["broker"]._on_app_teardown()
                            or self.stop_calls
                        ).result(timeout=2)
                    self.worker_observations.append(observed)
                    return False
                self.active = False
                self.handle = None
                return True
            finally:
                self.concurrent -= 1

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 2
    owner._reusable_cleanup_retry_delays_ms = (0, 1)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    holder["broker"] = broker
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(75)

    assert manager.worker_observations == [1]
    assert manager.stop_calls == 2
    assert manager.max_concurrent == 1
    assert broker.pending_count == baseline


@pytest.mark.parametrize("delivery", ["retry", "teardown"])
@pytest.mark.parametrize(
    "failure_kind", ["false", "keyboard", "system-exit", "hostile"]
)
def test_worker_demand_accepted_during_last_running_attempt_stays_in_burst(
    qapp, delivery, failure_kind
):
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False
            self.worker_deliveries = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    delivered = executor.submit(
                        holder["deliver_from_worker"]
                    ).result(timeout=2)
                assert delivered is True
                self.worker_deliveries += 1
            if self.allow_stop:
                self.active = False
                self.handle = None
                return True
            if failure_kind == "keyboard":
                raise KeyboardInterrupt()
            if failure_kind == "system-exit":
                raise SystemExit(109)
            if failure_kind == "hostile":
                raise _HostileStateError()
            return False

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count

    def deliver_from_worker():
        if delivery == "retry":
            return broker.retry_pending() == baseline + 1
        broker._on_app_teardown()
        return True

    holder["deliver_from_worker"] = deliver_from_worker
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(75)

    assert manager.worker_deliveries == 1
    assert manager.stop_calls == 1
    token = owner._reusable_detached_cleanup_tokens[
        ("shortcut", id(manager))
    ]
    assert token._state == token.IDLE
    QTest.qWait(50)
    assert manager.stop_calls == 1

    manager.allow_stop = True
    assert broker.retry_pending() == baseline + 1
    QTest.qWait(50)
    assert manager.stop_calls == 2
    assert broker.pending_count == baseline


@pytest.mark.parametrize("delivery", ["retry", "teardown"])
@pytest.mark.parametrize(
    "failure_kind", ["false", "keyboard", "system-exit", "hostile"]
)
def test_worker_demand_accepted_while_scheduled_stays_in_that_burst(
    qapp, monkeypatch, delivery, failure_kind
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False

        def stop(self):
            self.stop_calls += 1
            if self.allow_stop:
                self.active = False
                self.handle = None
                return True
            if failure_kind == "keyboard":
                raise KeyboardInterrupt()
            if failure_kind == "system-exit":
                raise SystemExit(113)
            if failure_kind == "hostile":
                raise _HostileStateError()
            return False

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(manager))
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    token = owner._reusable_detached_cleanup_tokens[exact_key]
    assert token._state == token.SCHEDULED
    generation = token._burst_generation
    request_sequence_before_worker = token._request_sequence
    delivered_receipts = []
    original_deliver_accepted = type(token).deliver_accepted_demand

    def observe_deliver_accepted(token_instance, acceptance, *, teardown=False):
        delivered_receipts.append(
            (
                acceptance,
                teardown,
                QThread.currentThread() == token_instance._affinity_thread,
            )
        )
        return original_deliver_accepted(
            token_instance, acceptance, teardown=teardown
        )

    monkeypatch.setattr(
        type(token), "deliver_accepted_demand", observe_deliver_accepted
    )

    def deliver_from_worker():
        if delivery == "retry":
            return broker.retry_pending() == baseline + 1
        broker._on_app_teardown()
        return True

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(deliver_from_worker).result(timeout=2) is True

    assert manager.stop_calls == 0
    assert token._state == token.SCHEDULED
    with broker._lock:
        entry = broker._registry[exact_key]
        acceptance = entry.demand_acceptance
        assert entry.demand_pending is True
    assert acceptance == (
        generation,
        request_sequence_before_worker + 1,
    )
    assert token._request_sequence == request_sequence_before_worker + 1
    assert delivered_receipts == []
    # Settle the one-attempt burst before the queued broker delivery. The
    # earlier queued token signal is then stale and cannot run another stop.
    token._attempt(generation)
    assert manager.stop_calls == 1
    assert token._state == token.IDLE
    QTest.qWait(50)
    assert delivered_receipts == [
        (acceptance, delivery == "teardown", True)
    ]
    with broker._lock:
        assert entry.demand_pending is False
        assert entry.demand_acceptance is None
    assert token._coalesced_through_sequence == acceptance[1]
    assert token._teardown_pending is (delivery == "teardown")
    assert manager.stop_calls == 1
    assert token._state == token.IDLE

    manager.allow_stop = True
    assert broker.retry_pending() == baseline + 1
    QTest.qWait(50)
    assert manager.stop_calls == 2
    assert broker.pending_count == baseline
    assert len(delivered_receipts) == 1


def test_real_app_teardown_rebuilds_provisional_after_owner_gc_before_exec_returns():
    repo_root = Path(__file__).resolve().parents[2]
    script = r'''
import gc
import os
import sys
import types
import weakref
from threading import RLock
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
preferences = types.ModuleType("base.analysis_warning_preferences")
preferences.is_uncalibrated_microphone_warning_suppressed = lambda: False
preferences.save_uncalibrated_microphone_warning_suppressed = lambda _value: None
sys.modules[preferences.__name__] = preferences

from PyQt5 import sip
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QApplication
import ui.sequence.sequence_resource_lifecycle_controller as module
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_resource_lifecycle_controller import (
    SequenceResourceLifecycleController,
)

app = QApplication([])
original_token_type = module._DetachedReusableCleanupToken

class FailOnceToken(original_token_type):
    construction_calls = 0
    def __new__(cls, *args, **kwargs):
        cls.construction_calls += 1
        if cls.construction_calls == 1:
            raise SystemExit(105)
        return super().__new__(cls)

module._DetachedReusableCleanupToken = FailOnceToken

class Owner(QObject):
    pass

class Target:
    def __init__(self):
        self.active = True
        self.handle = object()
        self.stop_calls = 0
    def stop(self):
        self.stop_calls += 1
        self.active = False
        self.handle = None
        return True

target = Target()
owner = Owner()
owner._shortcut_mgr = target
owner.hw_manager = SimpleNamespace(active=False, handle=None, stop=lambda: True)
owner.tcp_resource_port = SimpleNamespace(model=SequenceTriggerModel(tcp_enabled=False))
owner._reusable_resource_lock = RLock()
owner._reusable_resource_epoch = 0
owner._reusable_resource_state = "SUSPENDED"
owner._reusable_resource_journal = {}
owner._reusable_cleanup_retry_limit = 1
owner._reusable_cleanup_retry_delays_ms = (0,)
module.SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
broker = module.SequenceResourceLifecycleController._reusable_detached_broker(owner)
baseline = broker.pending_count
assert module.SequenceResourceLifecycleController._reserve_detached_reusable_target(owner, "shortcut", target)
module.SequenceResourceLifecycleController._schedule_detached_reusable_retry(owner, "shortcut", target)
assert FailOnceToken.construction_calls == 1
owner_ref = weakref.ref(owner)
sip.delete(owner)
del owner
gc.collect()
assert owner_ref() is None

QTimer.singleShot(0, app.quit)
app.exec_()
assert FailOnceToken.construction_calls == 2
assert target.stop_calls == 1
assert target.active is False
assert broker.pending_count == baseline
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_reusable_cleanup_round_reopens_exhausted_broker_token(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False

        def stop(self):
            self.stop_calls += 1
            if not self.allow_stop:
                return False
            self.active = False
            self.handle = None
            return True

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(50)
    assert broker.pending_count == baseline + 1

    manager.allow_stop = True
    with owner._reusable_resource_lock:
        generation = SequenceResourceLifecycleController._begin_reusable_cleanup_round_locked(
            owner, "SUSPENDED"
        )
    assert SequenceResourceLifecycleController._run_queued_reusable_cleanup(
        owner, generation=generation
    ) is False
    QTest.qWait(50)
    assert manager.active is False
    assert broker.pending_count == baseline


def test_app_broker_rejects_same_target_from_second_window(qapp):
    import ui.sequence.sequence_resource_lifecycle_controller as sequence_widget_module
    from ui.sequence.sequence_widget import SequenceWindow

    target = object()
    first = _loop5_reusable_owner(target)
    second = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(first)
    baseline = broker.pending_count
    assert broker.parent() is qapp
    assert not hasattr(
        sequence_widget_module, "_REUSABLE_DETACHED_BROKER"
    )

    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        first, "shortcut", target
    ) is True
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        second, "shortcut", target
    ) is False
    assert broker.pending_count == baseline + 1
    assert second._reusable_detached_pending_stops == {}

    SequenceResourceLifecycleController._release_detached_reusable_target(
        first, "shortcut", target
    )
    assert broker.pending_count == baseline


@pytest.mark.parametrize(
    "stop_outcome",
    [False, KeyboardInterrupt(), SystemExit(91)],
)
def test_app_broker_teardown_never_assumes_a_failed_stop_succeeded(
    qapp, stop_outcome
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Target:
        active = True
        handle = object()

        def stop(self):
            if isinstance(stop_outcome, BaseException):
                raise stop_outcome
            return stop_outcome

    target = Target()
    owner = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count

    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    assert broker.pending_count == baseline + 1

    broker._on_app_teardown()

    assert broker.pending_count == baseline + 1
    SequenceResourceLifecycleController._release_detached_reusable_target(
        owner, "shortcut", target
    )
    assert broker.pending_count == baseline


def test_app_broker_teardown_verified_stop_releases_owner_journal(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class Target:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.allow_stop = False

        def stop(self):
            if not self.allow_stop:
                return False
            self.active = False
            self.handle = None
            return True

    target = Target()
    owner = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", target
    )
    QTest.qWait(50)
    assert broker.pending_count == baseline + 1

    target.allow_stop = True
    broker._on_app_teardown()
    QTest.qWait(50)

    assert broker.pending_count == baseline
    assert owner._reusable_detached_pending_stops == {}
    assert owner._reusable_detached_cleanup_tokens == {}


def test_detached_token_serializes_owner_broker_and_teardown_round_requests(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    entered = Event()
    release = Event()
    concurrency_lock = RLock()

    class NonReentrantManager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.concurrent = 0
            self.max_concurrent = 0
            self.allow_stop = False

        def stop(self):
            with concurrency_lock:
                self.stop_calls += 1
                call_index = self.stop_calls
                self.concurrent += 1
                self.max_concurrent = max(
                    self.max_concurrent, self.concurrent
                )
            try:
                if call_index == 1:
                    entered.set()
                    assert release.wait(2)
                if not self.allow_stop:
                    return False
                self.active = False
                self.handle = None
                return True
            finally:
                with concurrency_lock:
                    self.concurrent -= 1

    manager = NonReentrantManager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_delays_ms = (0, 1, 1)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )

    def request_from_owner():
        assert entered.wait(2)
        return SequenceResourceLifecycleController._retry_detached_reusable_stops(owner)

    def request_from_broker():
        assert entered.wait(2)
        return broker.retry_pending()

    def request_from_teardown():
        assert entered.wait(2)
        broker._on_app_teardown()

    with ThreadPoolExecutor(max_workers=3) as executor:
        owner_request = executor.submit(request_from_owner)
        broker_request = executor.submit(request_from_broker)
        teardown_request = executor.submit(request_from_teardown)
        Timer(0.1, release.set).start()
        QTest.qWait(250)
        owner_request.result(timeout=2)
        broker_request.result(timeout=2)
        teardown_request.result(timeout=2)

    assert manager.max_concurrent == 1
    assert manager.stop_calls == owner._reusable_cleanup_retry_limit
    assert broker.pending_count == baseline + 1
    token = owner._reusable_detached_cleanup_tokens[
        ("shortcut", id(manager))
    ]
    assert token._state == token.IDLE
    exhausted_calls = manager.stop_calls
    QTest.qWait(25)
    assert manager.stop_calls == exhausted_calls

    token._attempt(token._burst_generation - 1)
    assert manager.stop_calls == exhausted_calls

    manager.allow_stop = True
    assert broker.retry_pending() == 1
    QTest.qWait(50)
    assert broker.pending_count == baseline


@pytest.mark.parametrize(
    "failure",
    [
        False,
        RuntimeError("stop"),
        KeyboardInterrupt(),
        SystemExit(93),
        _HostileStateError(),
    ],
)
def test_detached_reentrant_requests_do_not_reset_current_burst_budget(
    qapp, failure
):
    from ui.sequence.sequence_widget import SequenceWindow

    holder = {}

    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.allow_stop = False

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls <= holder["owner"]._reusable_cleanup_retry_limit:
                holder["broker"].retry_pending()
                SequenceResourceLifecycleController._retry_detached_reusable_stops(holder["owner"])
            if self.allow_stop:
                self.active = False
                self.handle = None
                return True
            if isinstance(failure, BaseException):
                raise failure
            return failure

    manager = Manager()
    owner = _loop5_reusable_owner(manager)
    owner._reusable_cleanup_retry_delays_ms = (0, 1, 1)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    holder.update(owner=owner, broker=broker)
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", manager
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", manager
    )
    QTest.qWait(75)

    assert manager.stop_calls == owner._reusable_cleanup_retry_limit
    token = owner._reusable_detached_cleanup_tokens[
        ("shortcut", id(manager))
    ]
    assert token._state == token.IDLE
    QTest.qWait(25)
    assert manager.stop_calls == owner._reusable_cleanup_retry_limit

    manager.allow_stop = True
    assert broker.retry_pending() == 1
    QTest.qWait(50)
    assert manager.stop_calls == owner._reusable_cleanup_retry_limit + 1
    assert broker.pending_count == baseline


@pytest.mark.parametrize(
    "teardown_failure",
    [False, KeyboardInterrupt(), SystemExit(95)],
)
def test_app_broker_teardown_attempts_idle_token_synchronously_and_retains_failure(
    qapp, teardown_failure
):
    from ui.sequence.sequence_widget import SequenceWindow

    class Target:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.outcome = False

        def stop(self):
            self.stop_calls += 1
            outcome = self.outcome
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                self.active = False
                self.handle = None
            return outcome

    target = Target()
    owner = _loop5_reusable_owner(target)
    owner._reusable_cleanup_retry_limit = 1
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    SequenceResourceLifecycleController._schedule_detached_reusable_retry(
        owner, "shortcut", target
    )
    QTest.qWait(30)
    assert target.stop_calls == 1

    target.outcome = teardown_failure
    broker._on_app_teardown()

    assert target.stop_calls == 2
    assert broker.pending_count == baseline + 1
    token = owner._reusable_detached_cleanup_tokens[
        ("shortcut", id(target))
    ]
    assert token._state == token.IDLE
    assert token._teardown_pending is True

    target.outcome = True
    assert broker.retry_pending() == 1
    QTest.qWait(30)
    assert broker.pending_count == baseline


def test_real_qapplication_quit_stops_idle_detached_token_before_exec_returns():
    repo_root = Path(__file__).resolve().parents[2]
    script = r'''
import os
import sys
import types
from threading import RLock
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
preferences = types.ModuleType("base.analysis_warning_preferences")
preferences.is_uncalibrated_microphone_warning_suppressed = lambda: False
preferences.save_uncalibrated_microphone_warning_suppressed = lambda _value: None
sys.modules[preferences.__name__] = preferences

from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QApplication
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_resource_lifecycle_controller import (
    SequenceResourceLifecycleController,
)
from ui.sequence.sequence_widget import SequenceWindow

app = QApplication([])

class Owner(QObject):
    pass

class Target:
    def __init__(self):
        self.active = True
        self.handle = object()
        self.stop_calls = 0
        self.allow_stop = False

    def stop(self):
        self.stop_calls += 1
        if not self.allow_stop:
            return False
        self.active = False
        self.handle = None
        return True

target = Target()
owner = Owner()
owner._shortcut_mgr = target
owner.hw_manager = SimpleNamespace(active=False, handle=None, stop=lambda: True)
owner.tcp_resource_port = SimpleNamespace(model=SequenceTriggerModel(tcp_enabled=False))
owner._reusable_resource_lock = RLock()
owner._reusable_resource_epoch = 0
owner._reusable_resource_state = "SUSPENDED"
owner._reusable_resource_journal = {}
owner._reusable_cleanup_retry_limit = 1
owner._reusable_cleanup_retry_delays_ms = (0,)
SequenceResourceLifecycleController._reusable_resource_identity_lock(owner)
broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
baseline = broker.pending_count
assert SequenceResourceLifecycleController._reserve_detached_reusable_target(owner, "shortcut", target)
SequenceResourceLifecycleController._schedule_detached_reusable_retry(owner, "shortcut", target)

def quit_after_exhaustion():
    assert target.stop_calls == 1
    target.allow_stop = True
    app.quit()

QTimer.singleShot(25, quit_after_exhaustion)
app.exec_()
assert target.stop_calls == 2, target.stop_calls
assert target.active is False
assert broker.pending_count == baseline
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_detached_exact_keys_reject_same_target_for_independent_resources(qapp):
    import gc
    from ui.sequence.sequence_widget import SequenceWindow

    class Target:
        pass

    target = Target()
    target_ref = weakref.ref(target)
    owner = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    shortcut_key = ("shortcut", id(target))
    tcp_key = ("tcp", id(target))

    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    assert broker.pending_count == baseline + 1
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "tcp", target
    ) is False
    assert set(owner._reusable_detached_pending_stops) == {shortcut_key}
    assert set(owner._reusable_detached_pending_resources) == {shortcut_key}
    assert tcp_key not in owner._reusable_detached_cleanup_tokens
    assert broker.pending_count == baseline + 1

    assert SequenceResourceLifecycleController._release_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    assert broker.pending_count == baseline

    owner._shortcut_mgr = None
    del target
    gc.collect()
    assert target_ref() is None


def test_detached_same_exact_key_rejects_second_window_without_orphan(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    target = object()
    first = _loop5_reusable_owner(target)
    second = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(first)
    baseline = broker.pending_count

    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        first, "shortcut", target
    ) is True
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        second, "shortcut", target
    ) is False
    assert broker.pending_count == baseline + 1
    assert second._reusable_detached_pending_stops == {}

    assert SequenceResourceLifecycleController._release_detached_reusable_target(
        first, "shortcut", target
    ) is True
    assert broker.pending_count == baseline


def test_detached_exact_key_release_and_reserve_have_no_orphan(
    qapp, monkeypatch
):
    from ui.sequence.sequence_widget import SequenceWindow

    target = object()
    owner = _loop5_reusable_owner(target)
    broker = SequenceResourceLifecycleController._reusable_detached_broker(owner)
    baseline = broker.pending_count
    exact_key = ("shortcut", id(target))
    broker_released = Event()
    allow_release_return = Event()
    reserve_started = Event()
    reserve_done = Event()
    original_release = broker.release_provisional

    def release_with_barrier(key, candidate):
        released = original_release(key, candidate)
        broker_released.set()
        assert allow_release_return.wait(2)
        return released

    monkeypatch.setattr(broker, "release_provisional", release_with_barrier)
    assert SequenceResourceLifecycleController._reserve_detached_reusable_target(
        owner, "shortcut", target
    ) is True

    with ThreadPoolExecutor(max_workers=2) as executor:
        releasing = executor.submit(
            SequenceResourceLifecycleController._release_detached_reusable_target,
            owner,
            "shortcut",
            target,
        )
        assert broker_released.wait(2)

        def reserve_again():
            reserve_started.set()
            try:
                return SequenceResourceLifecycleController._reserve_detached_reusable_target(
                    owner, "shortcut", target
                )
            finally:
                reserve_done.set()

        reserving = executor.submit(reserve_again)
        assert reserve_started.wait(2)
        assert reserve_done.wait(0.1) is False
        allow_release_return.set()
        assert releasing.result(timeout=2) is True
        assert reserving.result(timeout=2) is True

    assert owner._reusable_detached_pending_stops == {exact_key: target}
    assert broker.pending_count == baseline + 1
    assert SequenceResourceLifecycleController._release_detached_reusable_target(
        owner, "shortcut", target
    ) is True
    assert broker.pending_count == baseline


def test_final_cleanup_cancellation_releases_bounded_pending_before_delivery():
    pending = object()
    owner = SimpleNamespace(
        _reusable_resource_lock=RLock(),
        _reusable_cleanup_generation=7,
        _reusable_cleanup_attempt=2,
        _reusable_cleanup_event_pending=True,
        _reusable_resource_journal={
            "shortcut": {"pending_stops": {id(pending): pending}}
        },
    )

    SequenceResourceLifecycleController._cancel_reusable_cleanup_events(
        owner, release_pending=True
    )

    assert owner._reusable_cleanup_generation == 8
    assert owner._reusable_cleanup_attempt == 0
    assert owner._reusable_cleanup_event_pending is False
    assert owner._reusable_resource_journal["shortcut"]["pending_stops"] == {}


def _loop5_canonical_owner(controller_state="ACTIVE"):
    class Owner(QObject):
        pass

    controller = SimpleNamespace(
        _lifecycle_lock=RLock(),
        _lifecycle_state=controller_state,
        _resource_identity_epoch=0,
        _tcp_stop_journal={},
        _tcp_stop_completed_handles={},
    )
    owner = Owner()
    owner._reusable_resource_lock = RLock()
    owner._reusable_resource_epoch = 0
    owner._reusable_resource_state = "ACTIVE"
    owner._tcp_resource_port = controller
    return owner, controller


def test_canonical_class_mirror_routes_external_writes_to_all_live_owner_epochs_and_gc(
    qapp,
):
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    first_owner, first_controller = _loop5_canonical_owner()
    second_owner, second_controller = _loop5_canonical_owner()
    assert _CANONICAL_TCP_MIRROR_STATE.register(first_owner) is True
    assert _CANONICAL_TCP_MIRROR_STATE.register(second_owner) is True
    baseline = _CANONICAL_TCP_MIRROR_STATE.owner_count
    server = SimpleNamespace(running=True)

    SequenceWindow.tcp_server = server
    assert SequenceWindow.tcp_server is server
    assert _CANONICAL_TCP_MIRROR_STATE.read() is server
    assert first_controller._resource_identity_epoch == 1
    assert second_controller._resource_identity_epoch == 1
    assert first_owner._reusable_resource_epoch == 1
    assert second_owner._reusable_resource_epoch == 1

    SequenceWindow.tcp_server = None
    assert _CANONICAL_TCP_MIRROR_STATE.read() is None
    first_ref = weakref.ref(first_owner)
    del first_owner
    gc.collect()
    assert first_ref() is None
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline - 1


def test_canonical_owner_unregister_is_token_safe_and_preserves_other_identity(qapp):
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    baseline = _CANONICAL_TCP_MIRROR_STATE.owner_count
    owner, _controller = _loop5_canonical_owner()
    token = _CANONICAL_TCP_MIRROR_STATE.register_owner(owner)
    server = SimpleNamespace(running=True)
    assert token is not None
    assert _CANONICAL_TCP_MIRROR_STATE.write(server) is True

    assert _CANONICAL_TCP_MIRROR_STATE.unregister(token) is True
    assert _CANONICAL_TCP_MIRROR_STATE.read() is server
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline

    replacement = _CANONICAL_TCP_MIRROR_STATE.register_owner(owner)
    assert replacement is not None and replacement != token
    assert _CANONICAL_TCP_MIRROR_STATE.unregister(token) is False
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline + 1
    assert _CANONICAL_TCP_MIRROR_STATE.unregister(owner) is True
    assert _CANONICAL_TCP_MIRROR_STATE.unregister(owner) is True
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline
    SequenceWindow.tcp_server = None


def test_lifecycle_tcp_mirror_port_rejects_retired_writer_before_mutation(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    SequenceWindow.tcp_server = None

    class Bus:
        @staticmethod
        def publish_resource_lifecycle(_request):
            return True

        @staticmethod
        def close_workflow_continuation_dispatcher():
            return True

    owner = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(owner)
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner),
        lifecycle_bus=Bus(),
        parent=owner,
    )
    owner.resource_lifecycle_controller = controller
    first = object()
    late = object()
    assert owner._set_tcp_mirror_identity(first) is True
    assert owner._get_tcp_mirror_identity() is first

    assert controller._disconnect_trigger_inputs(1) is True
    assert owner._set_tcp_mirror_identity(late) is False
    assert SequenceWindow.tcp_server is first

    SequenceWindow.tcp_server = None


def test_lifecycle_tcp_mirror_port_requires_exact_registration_token(qapp):
    import copy
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    owner = QWidget()
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    registered_token = controller._tcp_mirror_owner_token
    forged_equal_token = copy.copy(registered_token)
    assert forged_equal_token == registered_token
    assert forged_equal_token is not registered_token

    candidate = object()
    assert _CANONICAL_TCP_MIRROR_STATE.write_registered(
        controller, forged_equal_token, candidate
    ) is False
    assert SequenceWindow.tcp_server is None

    assert controller.write_tcp_mirror_identity(candidate) is True
    assert controller.read_tcp_mirror_identity() is candidate
    assert _CANONICAL_TCP_MIRROR_STATE.unregister(
        controller, registered_token
    ) is True
    SequenceWindow.tcp_server = None


@pytest.mark.parametrize(
    "controller_state", ["DISCONNECTING", "FINALIZING", "INACTIVE"]
)
def test_lifecycle_tcp_mirror_port_preserves_clear_and_same_identity_semantics(
    qapp, controller_state
):
    from ui.sequence.sequence_widget import SequenceWindow

    SequenceWindow.tcp_server = None
    owner = QWidget()
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    trigger = SimpleNamespace(
        _lifecycle_lock=RLock(),
        _lifecycle_state="ACTIVE",
        _resource_identity_epoch=0,
        _tcp_stop_journal={},
        _tcp_stop_completed_handles={},
    )
    controller.tcp_resource_port = trigger
    server = object()
    assert controller.write_tcp_mirror_identity(server) is True

    trigger._lifecycle_state = controller_state
    assert controller.write_tcp_mirror_identity(server) is True
    assert controller.write_tcp_mirror_identity(object()) is False
    assert controller.read_tcp_mirror_identity() is server
    assert controller.write_tcp_mirror_identity(None) is True
    assert controller.read_tcp_mirror_identity() is None


def test_lifecycle_tcp_mirror_port_allows_hidden_suspend_reuse_and_hostile_identity(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    class HostileServer:
        def __eq__(self, _other):
            raise AssertionError("equality must not be observed")

        def __hash__(self):
            raise AssertionError("hash must not be observed")

        def __str__(self):
            raise AssertionError("text must not be observed")

    SequenceWindow.tcp_server = None
    owner = QWidget()
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    controller._reusable_resource_state = "SUSPENDED"
    controller._tcp_resource_port = SimpleNamespace(
        _lifecycle_lock=RLock(),
        _lifecycle_state="INACTIVE",
        _resource_identity_epoch=0,
        _tcp_stop_journal={},
        _tcp_stop_completed_handles={},
    )
    server = HostileServer()
    assert controller.write_tcp_mirror_identity(server) is True
    assert controller.write_tcp_mirror_identity(server) is True
    assert controller.read_tcp_mirror_identity() is server
    assert controller.write_tcp_mirror_identity(None) is True


def test_lifecycle_tcp_mirror_ports_isolate_retired_window_from_active_window(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    SequenceWindow.tcp_server = None
    first_owner = QWidget()
    second_owner = QWidget()
    first = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(first_owner), parent=first_owner
    )
    second = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(second_owner), parent=second_owner
    )
    initial = object()
    replacement = object()
    assert first.write_tcp_mirror_identity(initial) is True
    assert second.read_tcp_mirror_identity() is initial

    first_token = first._tcp_mirror_owner_token
    from ui.sequence.sequence_widget import _CANONICAL_TCP_MIRROR_STATE

    assert _CANONICAL_TCP_MIRROR_STATE.unregister(first, first_token) is True
    assert first.write_tcp_mirror_identity(replacement) is False
    assert first.read_tcp_mirror_identity() is None
    assert second.write_tcp_mirror_identity(replacement) is True
    assert second.read_tcp_mirror_identity() is replacement

    assert _CANONICAL_TCP_MIRROR_STATE.unregister(
        second, second._tcp_mirror_owner_token
    ) is True
    SequenceWindow.tcp_server = None


def test_lifecycle_tcp_mirror_port_rejects_native_deleted_owner(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import SequenceWindow

    SequenceWindow.tcp_server = None
    owner = QWidget()
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    port = controller.tcp_mirror_port
    sip.delete(controller)
    qapp.processEvents()

    assert port.write(object()) is False
    assert port.read() is None
    assert SequenceWindow.tcp_server is None


def test_lifecycle_tcp_mirror_retirement_and_write_are_atomic(qapp):
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    owner = QWidget()
    controller = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    initial = object()
    candidate = object()
    assert controller.write_tcp_mirror_identity(initial) is True
    token = controller._tcp_mirror_owner_token
    start = Event()

    def retire():
        start.wait(timeout=2)
        return _CANONICAL_TCP_MIRROR_STATE.unregister(controller, token)

    def write():
        start.wait(timeout=2)
        return controller.write_tcp_mirror_identity(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        retirement = executor.submit(retire)
        late_write = executor.submit(write)
        start.set()
        retired = retirement.result(timeout=2)
        written = late_write.result(timeout=2)

    assert retired is True
    assert _CANONICAL_TCP_MIRROR_STATE.read() is (
        candidate if written else initial
    )
    assert controller.write_tcp_mirror_identity(object()) is False
    SequenceWindow.tcp_server = None


def test_canonical_owner_native_delete_unregisters_without_strong_retention(qapp):
    from PyQt5 import sip
    from ui.sequence.sequence_widget import _CANONICAL_TCP_MIRROR_STATE

    baseline = _CANONICAL_TCP_MIRROR_STATE.owner_count
    owner, _controller = _loop5_canonical_owner()
    owner_ref = weakref.ref(owner)
    assert _CANONICAL_TCP_MIRROR_STATE.register_owner(owner) is not None
    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline + 1

    sip.delete(owner)
    qapp.processEvents()

    assert _CANONICAL_TCP_MIRROR_STATE.owner_count == baseline
    del owner
    gc.collect()
    assert owner_ref() is None


@pytest.mark.parametrize("controller_state", ["DISCONNECTING", "FINALIZING", "INACTIVE"])
def test_canonical_class_active_write_is_rejected_atomically_by_any_stopping_owner(
    qapp, controller_state
):
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    active_owner, active_controller = _loop5_canonical_owner("ACTIVE")
    rejecting_owner, rejecting_controller = _loop5_canonical_owner(
        controller_state
    )
    _CANONICAL_TCP_MIRROR_STATE.register(active_owner)
    _CANONICAL_TCP_MIRROR_STATE.register(rejecting_owner)
    candidate = SimpleNamespace(running=True)

    SequenceWindow.tcp_server = candidate
    assert SequenceWindow.tcp_server is None
    assert rejecting_controller._resource_identity_epoch == 1
    assert active_controller._tcp_stop_journal == {}


def test_canonical_class_write_in_trigger_final_review_gap_invalidates_epoch_cas(
    qapp, monkeypatch
):
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_trigger_view import SequenceTriggerView
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    view = SequenceTriggerView()
    controller = SequenceTriggerController(
        SequenceTriggerModel(),
        view,
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        shortcut_manager=SimpleNamespace(active=False, stop=lambda: True),
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
        tcp_mirror_getter=lambda: SequenceWindow.tcp_server,
        tcp_mirror_setter=_CANONICAL_TCP_MIRROR_STATE.write,
    )
    owner, _unused_controller = _loop5_canonical_owner()
    owner._tcp_resource_port = controller
    _CANONICAL_TCP_MIRROR_STATE.register(owner)
    original_review = controller._disconnect_resources_stable
    late = SimpleNamespace(running=True)

    def write_after_review():
        stable = original_review()
        SequenceWindow.tcp_server = late
        assert SequenceWindow.tcp_server is None
        return stable

    monkeypatch.setattr(
        controller, "_disconnect_resources_stable", write_after_review
    )
    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"

    monkeypatch.setattr(
        controller, "_disconnect_resources_stable", original_review
    )
    assert controller.disconnect() is True


def test_canonical_holder_and_controller_lock_order_has_no_thread_deadlock(qapp):
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_trigger_view import SequenceTriggerView
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    controller = SequenceTriggerController(
        SequenceTriggerModel(),
        SequenceTriggerView(),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        tcp_mirror_getter=lambda: SequenceWindow.tcp_server,
        tcp_mirror_setter=_CANONICAL_TCP_MIRROR_STATE.write,
    )
    owner, _unused_controller = _loop5_canonical_owner()
    owner._tcp_resource_port = controller
    _CANONICAL_TCP_MIRROR_STATE.register(owner)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writes = [
            executor.submit(controller.tcp_mirror_setter, None),
            executor.submit(
                setattr,
                SequenceWindow,
                "tcp_server",
                SimpleNamespace(running=True),
            ),
        ]
        assert all(future.result(timeout=2) is not False for future in writes)


def test_canonical_owner_admission_reverse_class_write_is_rejected_without_deadlock(
    qapp,
):
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    nested = SimpleNamespace(running=True)

    class Controller:
        def __init__(self):
            self._lifecycle_lock = RLock()
            self._resource_identity_epoch = 0
            self.observed_during_callback = None

        def _admit_canonical_tcp_mirror_identity_locked(self, previous, current):
            self._resource_identity_epoch += 1
            SequenceWindow.tcp_server = nested
            self.observed_during_callback = SequenceWindow.tcp_server
            return True

    owner, _unused_controller = _loop5_canonical_owner()
    controller = Controller()
    owner._tcp_resource_port = controller
    _CANONICAL_TCP_MIRROR_STATE.register(owner)
    outer = SimpleNamespace(running=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_CANONICAL_TCP_MIRROR_STATE.write, outer)
        assert future.result(timeout=2) is True

    assert controller.observed_during_callback is None
    assert SequenceWindow.tcp_server is outer
    assert controller._resource_identity_epoch == 1


def test_model_observer_to_controller_and_canonical_lock_order_survives_pressure(qapp):
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_trigger_view import SequenceTriggerView
    from ui.sequence.sequence_widget import (
        SequenceWindow,
        _CANONICAL_TCP_MIRROR_STATE,
    )

    SequenceWindow.tcp_server = None
    model = SequenceTriggerModel(tcp_identity_outbox_limit=8)
    controller = SequenceTriggerController(
        model,
        SequenceTriggerView(),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        tcp_mirror_getter=lambda: SequenceWindow.tcp_server,
        tcp_mirror_setter=_CANONICAL_TCP_MIRROR_STATE.write,
    )
    owner, _unused_controller = _loop5_canonical_owner()
    owner._tcp_resource_port = controller
    _CANONICAL_TCP_MIRROR_STATE.register(owner)

    class Observer:
        def notified(self):
            SequenceWindow.tcp_server = model.tcp_server

    observer = Observer()
    token = model.subscribe_tcp_identity_observer(observer.notified)
    servers = [SimpleNamespace(running=True) for _index in range(32)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                model.activate_tcp_server,
                server,
                lifecycle_generation=index,
                server_token=f"pressure-{index}",
            )
            for index, server in enumerate(servers)
        ]
        assert all(future.result(timeout=2) is True for future in futures)

    assert model.drain_tcp_identity_outbox() == ()
    assert SequenceWindow.tcp_server in servers
    assert model.unsubscribe_tcp_identity_observer(token) is True


def test_reusable_disabled_resume_keeps_failed_replacement_stop_pending():
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self, active, stop_results):
            self.active = active
            self.handle = object() if active else None
            self.stop_results = list(stop_results)

        def stop(self):
            result = self.stop_results.pop(0)
            if result:
                self.active = False
                self.handle = None
            return result

        def start(self):
            self.active = True
            self.handle = object()
            return True

    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=Manager(False, []),
        hw_manager=Manager(False, []),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True

    replacement = Manager(True, [False, True])
    facade.shortcut_mgr = replacement
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    assert facade._reusable_resource_state == "RESUMING"
    assert facade._reusable_resource_snapshot is not None
    assert replacement in facade._reusable_resource_journal["shortcut"][
        "pending_stops"
    ].values()

    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert facade._reusable_resource_snapshot is None


@pytest.mark.parametrize(
    "error",
    [RuntimeError("state"), KeyboardInterrupt(), SystemExit(10), _HostileStateError()],
)
def test_reusable_state_observation_baseexceptions_are_retryable(error):
    from ui.sequence.sequence_widget import SequenceWindow

    class Manager:
        def __init__(self):
            self.raise_state = True
            self.physically_active = True

        @property
        def active(self):
            if self.raise_state:
                raise error
            return self.physically_active

        def stop(self):
            self.physically_active = False
            return True

        def start(self):
            self.physically_active = True
            return True

    manager = Manager()
    facade = SimpleNamespace(
        default_logger=_ResourceLogger(),
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False)
        ),
        shortcut_mgr=manager,
        hw_manager=SimpleNamespace(active=False, handle=None),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_resource_state="ACTIVE",
        _reusable_resource_journal={},
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is False
    assert manager in facade._reusable_resource_journal["shortcut"][
        "pending_stops"
    ].values()
    manager.raise_state = False
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade) is True


def test_show_event_records_failed_resume_and_retries_on_next_event(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    widget = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(widget)
    lifecycle = _attach_resource_lifecycle_owner(widget)
    resume_results = [False, True]
    lifecycle._resume_reusable_child_resources = lambda: resume_results.pop(0)
    widget.configuration_view = SimpleNamespace(
        present_missing_configuration_prompt=lambda *_args, **_kwargs: True
    )
    widget.sequence_config = {}

    SequenceWindow.showEvent(widget, QShowEvent())
    assert lifecycle._reusable_resume_pending is True
    SequenceWindow.showEvent(widget, QShowEvent())
    assert lifecycle._reusable_resume_pending is False


def test_reusable_child_tcp_resume_uses_pre_stop_trigger_model_snapshot():
    from ui.sequence.sequence_widget import SequenceWindow

    model = SequenceTriggerModel(tcp_enabled=True, tcp_host="127.0.0.1", tcp_port=9001)
    resumes = []

    class Trigger:
        def __init__(self):
            self.model = model

        def stop_tcp(self):
            model.tcp_enabled = False
            model.tcp_running = False

        def set_tcp_enabled(self, enabled, **kwargs):
            resumes.append((enabled, kwargs))
            model.tcp_enabled = bool(enabled)
            return True

    facade = SimpleNamespace(
        tcp_resource_port=Trigger(),
        shortcut_mgr=SimpleNamespace(
            active=False, stop=lambda: None, start=lambda: True
        ),
        hw_manager=SimpleNamespace(
            active=False, stop=lambda: None, start=lambda: True
        ),
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_suspend_completed=set(),
        _reusable_resume_completed=set(),
    )

    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade)
    assert model.tcp_enabled is False
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade)
    assert model.tcp_enabled is True
    assert resumes == [(True, {"host": "127.0.0.1", "port": 9001})]
    assert facade._reusable_resource_snapshot is None


def test_reusable_child_resume_retries_only_failed_manager_step():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []
    shortcut_results = [RuntimeError("first"), True]

    shortcut = SimpleNamespace(active=True)
    hardware = SimpleNamespace(active=True)

    def shortcut_stop():
        shortcut.active = False

    def shortcut_start():
        calls.append("shortcut")
        result = shortcut_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result:
            shortcut.active = True
        return result

    def hardware_stop():
        hardware.active = False

    def hardware_start():
        calls.append("hardware")
        hardware.active = True
        return True

    shortcut.stop = shortcut_stop
    shortcut.start = shortcut_start
    hardware.stop = hardware_stop
    hardware.start = hardware_start

    facade = SimpleNamespace(
        tcp_resource_port=SimpleNamespace(
            model=SequenceTriggerModel(tcp_enabled=False), stop_tcp=lambda: None
        ),
        shortcut_mgr=shortcut,
        hw_manager=hardware,
        _reusable_child_suspended=False,
        _reusable_resource_snapshot=None,
        _reusable_suspend_completed=set(),
        _reusable_resume_completed=set(),
    )
    assert SequenceResourceLifecycleController._suspend_reusable_child_resources(facade)
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is False
    assert SequenceResourceLifecycleController._resume_reusable_child_resources(facade) is True
    assert calls == ["shortcut", "hardware", "shortcut"]


def test_single_show_event_resumes_resources_and_preserves_configuration_prompt(qapp):
    from ui.sequence.sequence_widget import SequenceWindow

    widget = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(widget)
    lifecycle = _attach_resource_lifecycle_owner(widget)
    calls = []
    lifecycle._resume_reusable_child_resources = lambda: calls.append("resume") or True
    widget.configuration_view = SimpleNamespace(
        present_missing_configuration_prompt=lambda config, **kwargs: calls.append(
            ("prompt", config, kwargs)
        )
    )
    widget.sequence_config = [{"name": "demo"}]

    SequenceWindow.showEvent(widget, QShowEvent())
    assert calls == [
        "resume",
        ("prompt", [{"name": "demo"}], {"eligible": True}),
    ]
    assert inspect.getsource(SequenceWindow).count("def showEvent") == 1


def test_legacy_shutdown_binding_connects_abort_before_return(qapp):
    from main_window import MainWindow

    class LegacyFacade(QObject):
        shutdown_ready = pyqtSignal(object)
        shutdown_aborted = pyqtSignal(object)

        def request_application_shutdown(self, _generation):
            return True

    facade = LegacyFacade()
    window = _MainHarness(facade)
    window.on_shutdown_ready = MainWindow.on_shutdown_ready.__get__(window)
    window.on_shutdown_aborted = MainWindow.on_shutdown_aborted.__get__(window)
    _invoke_main("_initialize_application_shutdown_state", window)
    assert _invoke_main("_bind_sequence_shutdown_facade", window, facade)
    window._shutdown_active_generation = 9

    facade.shutdown_aborted.emit(ShutdownAborted(9))
    qapp.processEvents()
    assert window._shutdown_active_generation is None
    _invoke_main("closeEvent", window, _CloseEvent())
    assert window._shutdown_active_generation == 0


@pytest.mark.parametrize("error", [RuntimeError("service"), KeyboardInterrupt(), SystemExit(7)])
def test_shutdown_worker_service_baseexceptions_reach_shutdown_failure(tmp_path, error):
    controller, view, submissions, completed, _output, _spool = (
        _shutdown_export_with_target(tmp_path)
    )
    job, _attempt = submissions.pop()

    def fail_service(_job, _attempt_id):
        raise error

    worker = SequenceExportWorker(job, job.attempt_id, execute=fail_service)
    worker.failed.connect(controller.handle_worker_failed)
    worker.run()

    assert controller.model.shutdown_flush_failure_identity == (
        job.job_id,
        job.attempt_id,
    )
    assert view.shutdown_failures[-1][:3] == (
        31,
        job.job_id,
        job.attempt_id,
    )
    assert completed == []


def test_resource_cleanup_retries_only_unfinished_ordered_step():
    from ui.sequence.sequence_widget import SequenceWindow

    calls = []
    analysis_attempts = [RuntimeError("first"), True]

    def close_analysis():
        calls.append("analysis")
        result = analysis_attempts.pop(0)
        if isinstance(result, BaseException):
            raise result

    facade = SimpleNamespace(
        _shutdown_prepared_generation=None,
        _shutdown_cleanup_trace=[],
        _shutdown_cleanup_steps_completed=set(),
        _suspend_reusable_child_resources=(
            lambda: calls.append("trigger") or True
        ),
        _view=SimpleNamespace(
            close_analysis_windows=close_analysis,
            close_application_subwindows=(
                lambda: calls.append("dialogs")
            ),
        ),
    )

    assert not SequenceResourceLifecycleController._prepare_application_shutdown_resources(
        facade, 94
    )
    assert SequenceResourceLifecycleController._prepare_application_shutdown_resources(facade, 94)
    assert calls == ["trigger", "analysis", "analysis", "dialogs"]
    assert facade._shutdown_cleanup_trace == [
        "stop-trigger-resources",
        "close-analysis-windows",
    ]


def test_shutdown_ready_waits_for_reentrant_manager_replacement_cleanup():
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_widget import SequenceWindow

    class View:
        def present_tcp_state(self, _enabled):
            return None

        def close_dialogs(self):
            return None

    class Manager:
        def __init__(self, on_stop=None):
            self.active = True
            self.handle = object()
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    replacement = Manager()
    holder = {}
    old = Manager(
        lambda: setattr(holder["trigger"], "shortcut_manager", replacement)
    )
    trigger = SequenceTriggerController(
        SequenceTriggerModel(),
        View(),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        shortcut_manager=old,
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    holder["trigger"] = trigger
    resources = SimpleNamespace(
        _shutdown_prepared_generation=None,
        _shutdown_cleanup_trace=[],
        _shutdown_cleanup_steps_completed=set(),
        _suspend_reusable_child_resources=trigger.disconnect,
        _view=SimpleNamespace(
            close_analysis_windows=lambda: None,
            close_application_subwindows=lambda: None,
        ),
    )
    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.shutdown_generation = 101
    workflow.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    ready = []
    coordinator = SequenceShutdownCoordinator(
        workflow,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: (
            SequenceResourceLifecycleController._prepare_application_shutdown_resources(
                resources, generation
            )
        ),
    )
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "main", lambda event: ready.append(event) or True
    )

    assert coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(101)
    ) is False
    assert ready == []
    assert trigger.lifecycle_state == "DISCONNECTING"
    assert replacement.stop_calls == 0

    assert coordinator.retry_pending_shutdown() is True
    assert replacement.stop_calls == 1
    assert ready == [ShutdownReady(101)]


def test_shutdown_ready_waits_for_cross_resource_reinstall_cleanup():
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_widget import SequenceWindow

    class View:
        def present_tcp_state(self, _enabled):
            return None

        def close_dialogs(self):
            return None

    class Resource:
        client_address = None

        def __init__(self, active=True, on_stop=None):
            self.active = active
            self.running = active
            self.handle = object() if active else None
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.active = False
            self.running = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    replacement_shortcut = Resource()
    replacement_tcp = Resource()
    mirror = {"server": None}
    holder = {}

    def reinstall():
        trigger = holder["trigger"]
        trigger.shortcut_manager = replacement_shortcut
        holder["model"].activate_tcp_server(
            replacement_tcp,
            lifecycle_generation=trigger.lifecycle_generation,
            server_token="shutdown-cross-resource",
        )
        trigger.tcp_mirror_setter(replacement_tcp)

    model = SequenceTriggerModel()
    trigger = SequenceTriggerController(
        model,
        View(),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        shortcut_manager=Resource(False),
        hardware_manager=Resource(on_stop=reinstall),
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    holder.update(trigger=trigger, model=model)
    resources = SimpleNamespace(
        _shutdown_prepared_generation=None,
        _shutdown_cleanup_trace=[],
        _shutdown_cleanup_steps_completed=set(),
        _suspend_reusable_child_resources=trigger.disconnect,
        _view=SimpleNamespace(
            close_analysis_windows=lambda: None,
            close_application_subwindows=lambda: None,
        ),
    )
    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.shutdown_generation = 102
    workflow.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    ready = []
    coordinator = SequenceShutdownCoordinator(
        workflow,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: (
            SequenceResourceLifecycleController._prepare_application_shutdown_resources(
                resources, generation
            )
        ),
    )
    bus.register_workflow_continuation_recipient(
        "shutdown-ready", "main", lambda event: ready.append(event) or True
    )

    assert coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(102)
    ) is False
    assert ready == []
    assert trigger.lifecycle_state == "DISCONNECTING"
    assert replacement_shortcut.stop_calls == replacement_tcp.stop_calls == 0

    assert coordinator.retry_pending_shutdown() is True
    assert replacement_shortcut.stop_calls == replacement_tcp.stop_calls == 1
    assert ready == [ShutdownReady(102)]


@pytest.mark.parametrize("first_stop", [False, RuntimeError("tcp stop")])
def test_real_tcp_shutdown_failure_keeps_main_visible_and_handle_until_retry(
    qapp, first_stop
):
    from PyQt5.QtWidgets import QMainWindow
    from main_window import MainWindow
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_widget import SequenceWindow

    mirror = {"server": None}

    class Server:
        client_address = None

        def __init__(self, **_kwargs):
            self.stop_results = [first_stop, True]
            self.stop_calls = 0

        def start(self):
            return True

        def stop(self):
            self.stop_calls += 1
            result = self.stop_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    trigger_model = SequenceTriggerModel()
    trigger = SequenceTriggerController(
        trigger_model,
        SimpleNamespace(
            present_tcp_state=lambda _enabled: True,
            close_dialogs=lambda: True,
        ),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        tcp_server_factory=Server,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert trigger.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    server = trigger_model.tcp_server

    resource_facade = SimpleNamespace(
        _shutdown_prepared_generation=None,
        _shutdown_cleanup_trace=[],
        _shutdown_cleanup_steps_completed=set(),
        _suspend_reusable_child_resources=trigger.disconnect,
        _view=SimpleNamespace(
            close_analysis_windows=lambda: None,
            close_application_subwindows=lambda: None,
        ),
    )
    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.shutdown_generation = 0
    workflow.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    coordinator = SequenceShutdownCoordinator(
        workflow,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: (
            SequenceResourceLifecycleController._prepare_application_shutdown_resources(
                resource_facade, generation
            )
        ),
    )

    class Facade:
        shutdown_aborted = None

        def register_shutdown_ready_recipient(self, recipient, *, owner=None):
            bus.register_workflow_continuation_recipient(
                "shutdown-ready", "main", recipient, owner=owner
            )
            return True

        def request_application_shutdown(self, generation):
            QTimer.singleShot(
                0,
                lambda: coordinator.handle_shutdown_flush_completed(
                    ShutdownFlushCompleted(generation)
                ),
            )
            return True

        def raise_shutdown_progress(self, generation):
            return coordinator.raise_progress(generation)

    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            MainWindow._initialize_application_shutdown_state(self)
            self.sequence_window = Facade()

    window = Window()
    window.show()
    assert window.close() is False
    qapp.processEvents()
    assert window.isVisible() is True
    assert window._shutdown_active_generation == 0
    assert trigger.is_active is False
    assert trigger.lifecycle_state == "DISCONNECTING"
    assert trigger_model.tcp_server is server
    assert mirror["server"] is server

    assert window.close() is False
    qapp.processEvents()
    assert trigger.is_active is False
    assert trigger_model.tcp_server is None
    assert mirror["server"] is None

    # The repeated close restarts the event-driven cleanup round. The exact
    # successful cleanup publishes Ready and its guarded nested close exits.
    assert window.isVisible() is False
    assert coordinator.retry_pending_shutdown() is False
    assert trigger.is_active is False
    assert trigger_model.tcp_server is None
    assert mirror["server"] is None
    assert server.stop_calls == 2


def test_real_hid_close_failure_keeps_main_visible_until_exact_handle_retries(qapp):
    from PyQt5.QtWidgets import QMainWindow
    from base.unified_hid_device_manager import UnifiedHardwareManager
    from main_window import MainWindow
    from ui.sequence.sequence_trigger_controller import SequenceTriggerController
    from ui.sequence.sequence_widget import SequenceWindow

    class Handle:
        def __init__(self):
            self.results = [False, True]
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            return self.results.pop(0)

    handle = Handle()
    hardware = UnifiedHardwareManager()
    hardware.logger = SimpleNamespace(
        info=lambda *_args: None,
        warning=lambda *_args: None,
        error=lambda *_args: None,
    )
    hardware._scanner_enabled = True
    hardware.hid_handles["scanner"] = {"exact": handle}
    trigger_model = SequenceTriggerModel()
    trigger = SequenceTriggerController(
        trigger_model,
        SimpleNamespace(
            present_tcp_state=lambda _enabled: True,
            close_dialogs=lambda: True,
        ),
        start_publisher=lambda _event: None,
        configuration_generation_provider=lambda: 0,
        hardware_manager=hardware,
    )
    resources = SimpleNamespace(
        _shutdown_prepared_generation=None,
        _shutdown_cleanup_trace=[],
        _shutdown_cleanup_steps_completed=set(),
        _suspend_reusable_child_resources=trigger.disconnect,
        _view=SimpleNamespace(
            close_analysis_windows=lambda: None,
            close_application_subwindows=lambda: None,
        ),
    )
    bus = SequenceEventBus()
    workflow = SequenceWorkflowModel()
    workflow.shutdown_generation = 0
    workflow.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    coordinator = SequenceShutdownCoordinator(
        workflow,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: (
            SequenceResourceLifecycleController._prepare_application_shutdown_resources(
                resources, generation
            )
        ),
    )

    class Facade:
        shutdown_aborted = None

        def register_shutdown_ready_recipient(self, recipient, *, owner=None):
            bus.register_workflow_continuation_recipient(
                "shutdown-ready", "main", recipient, owner=owner
            )
            return True

        def request_application_shutdown(self, generation):
            QTimer.singleShot(
                0,
                lambda: coordinator.handle_shutdown_flush_completed(
                    ShutdownFlushCompleted(generation)
                ),
            )
            return True

        def raise_shutdown_progress(self, generation):
            return coordinator.raise_progress(generation)

    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            MainWindow._initialize_application_shutdown_state(self)
            self.sequence_window = Facade()

    window = Window()
    window.show()
    assert window.close() is False
    qapp.processEvents()
    assert window.isVisible() is True
    assert window._shutdown_active_generation == 0
    assert trigger.lifecycle_state == "DISCONNECTING"
    assert hardware.hid_handles["scanner"] == {"exact": handle}
    assert handle.close_calls == 1

    assert window.close() is False
    qapp.processEvents()
    assert trigger.lifecycle_state == "INACTIVE"
    assert hardware.hid_handles == {}
    assert window.isVisible() is False
    assert coordinator.retry_pending_shutdown() is False
    assert handle.close_calls == 2


def test_real_main_sequence_workflow_export_wiring_survives_startup_hide(qapp):
    from PyQt5.QtWidgets import QMainWindow
    from main_window import MainWindow
    from ui.sequence.sequence_widget import SequenceWindow

    class Trigger:
        is_active = True
        model = SequenceTriggerModel(tcp_enabled=False)

        def stop_tcp(self):
            return None

        def disconnect(self, _request=None):
            return True

    sequence = SequenceWindow.__new__(SequenceWindow)
    QWidget.__init__(sequence)
    sequence.sequence_event_bus = SequenceEventBus(sequence)
    lifecycle = _attach_resource_lifecycle_owner(sequence)
    SequenceWindow._initialize_shutdown_ready_release_port(sequence)
    lifecycle._reusable_child_suspended = False
    lifecycle._lightweight_cleanup_done = False
    sequence.trigger_controller = Trigger()
    sequence.shortcut_mgr = SimpleNamespace(
        active=False, handle=None, stop=lambda: None, start=lambda: None
    )
    sequence.hw_manager = SimpleNamespace(
        active=False, handle=None, stop=lambda: None, start=lambda: None
    )
    sequence._close_analysis_windows = lambda: None
    sequence.configuration_view = SimpleNamespace(
        present_missing_configuration_prompt=lambda *_args, **_kwargs: True
    )
    sequence.sequence_config = {}
    sequence.workflow_model = SequenceWorkflowModel()
    sequence.workflow_controller = SequenceWorkflowController(
        sequence.workflow_model, sequence.sequence_event_bus, parent=sequence
    )
    sequence.export_model = SequenceExportModel()
    sequence.export_controller = SequenceExportController(
        sequence.export_model,
        _ExportView(),
        bus=sequence.sequence_event_bus,
        submit_attempt=lambda *_args: None,
        parent=sequence,
    )
    sequence.shutdown_coordinator = SequenceShutdownCoordinator(
        sequence.workflow_model,
        sequence.sequence_event_bus,
        view=_ShutdownView(),
        cleanup_resources=lifecycle.complete_application_shutdown_before_ready,
        shutdown_ready=sequence.workflow_controller.handle_shutdown_ready,
        finalize_after_ready_ack=(
            lifecycle.complete_application_shutdown_after_ready_ack
        ),
        release_shutdown_close=sequence._release_staged_shutdown_close,
        parent=sequence,
    )
    sequence.sequence_event_bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "trigger",
        sequence.trigger_controller.disconnect,
    )
    sequence.sequence_event_bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "workflow",
        sequence.workflow_controller.disconnect,
        owner=sequence.workflow_controller,
    )
    sequence.sequence_event_bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "export",
        sequence.export_controller.disconnect,
        owner=sequence.export_controller,
    )
    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            MainWindow._initialize_application_shutdown_state(self)
            self.sequence_window = sequence
            self.setCentralWidget(sequence)

        def _close_all_subwindows(self):
            return None

    window = Window()
    sequence.close()
    sequence.show()
    qapp.processEvents()

    assert sequence.workflow_controller.handle_start(
        StartTestRequested("normal-start", "manual", "", False, 0)
    )
    session_id = sequence.workflow_model.active_session_id
    assert sequence.workflow_controller.handle_recording_failed(
        RecordingFailed(session_id, "test terminal")
    )
    assert sequence.workflow_model.phase is WorkflowPhase.IDLE

    window.show()
    assert window.close() is False
    qapp.processEvents()
    qapp.processEvents()
    qapp.processEvents()
    qapp.processEvents()
    qapp.processEvents()
    qapp.processEvents()
    assert not window.isVisible()
    assert sequence.export_model.shutdown_flush_terminal is True
    assert sequence.workflow_model.phase is WorkflowPhase.SHUTDOWN_READY
    assert lifecycle._shutdown_finalized_generation == 0
    assert lifecycle._shutdown_delivery_completed_generation == 0


class _DialogSignal:
    def connect(self, _slot):
        return None


class _RecoverableDialog:
    def __init__(self, stage=None):
        self.stage = stage
        self.buttonClicked = _DialogSignal()
        self.closed = False

    def _maybe(self, stage):
        if self.stage == stage:
            raise RuntimeError(f"{stage} failed")

    def setWindowModality(self, *_args):
        self._maybe("setup")

    def setIcon(self, *_args):
        return None

    def setWindowTitle(self, *_args):
        return None

    def setText(self, *_args):
        return None

    def addButton(self, *_args):
        return object()

    def setDefaultButton(self, *_args):
        return None

    def open(self):
        self._maybe("open")

    def close(self):
        self.closed = True

    def deleteLater(self):
        return None


@pytest.mark.parametrize("method", ["confirmation", "failure"])
@pytest.mark.parametrize("stage", ["factory", "setup", "open"])
def test_shutdown_dialog_identity_commits_only_after_successful_open(method, stage):
    calls = []

    def factory(_parent):
        calls.append(stage)
        if len(calls) == 1 and stage == "factory":
            raise RuntimeError("factory failed")
        return _RecoverableDialog(stage if len(calls) == 1 else None)

    view = SequenceExportView(failure_dialog_factory=factory)
    if method == "confirmation":
        invoke = lambda: view.show_shutdown_confirmation(91)
        identity = lambda: view._shutdown_confirmation_generation
    else:
        invoke = lambda: view.show_shutdown_failure(91, "job", "attempt", ("x",))
        identity = lambda: view._shutdown_failure_identity

    assert invoke() is False
    assert identity() is None
    assert invoke() is True
    assert identity() is not None


def test_shutdown_resource_order_is_idempotent_and_dispatchers_close_after_ready_ack():
    order = []
    bus = SequenceEventBus()
    original_close = bus.close_workflow_continuation_dispatcher
    bus.close_workflow_continuation_dispatcher = (
        lambda: order.append("dispatcher") or original_close()
    )
    view = SimpleNamespace(
        close_analysis_windows=lambda: order.append("analysis"),
        close_application_subwindows=lambda: order.append("dialogs"),
        remove_application_event_filter=lambda: None,
        disconnect_barcode_inputs=lambda: None,
    )
    lifecycle = SequenceResourceLifecycleController(
        view,
        SequenceResourceLifecycleModel(),
        lifecycle_bus=bus,
    )
    lifecycle._suspend_reusable_child_resources = (
        lambda: order.append("trigger") or True
    )
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "owners",
        lambda _request: order.append("owners") or True,
    )
    assert lifecycle.prepare_application_shutdown(2)
    assert lifecycle.prepare_application_shutdown(2)
    assert order == ["trigger", "analysis", "dialogs"]
    assert lifecycle.finalize_application_shutdown(2)
    assert lifecycle.finalize_application_shutdown(2)
    assert lifecycle.complete_application_shutdown_delivery(2)
    assert lifecycle.complete_application_shutdown_delivery(2)
    assert lifecycle.complete_application_shutdown_after_ready_ack(2)
    assert lifecycle.complete_application_shutdown_after_ready_ack(2)
    assert order == [
        "trigger",
        "analysis",
        "dialogs",
        "owners",
        "dispatcher",
    ]
    assert lifecycle._shutdown_cleanup_trace == [
        "stop-trigger-resources",
        "close-analysis-windows",
        "disconnect-sequence-owners",
    ]


def test_post_ready_dispatcher_retry_does_not_repeat_successful_close_step():
    order = []
    lifecycle_attempts = [False, True]

    bus = SequenceEventBus()
    original_close = bus.close_workflow_continuation_dispatcher
    bus.close_workflow_continuation_dispatcher = (
        lambda: order.append("dispatcher") or original_close()
    )
    original_lifecycle_close = bus.close_resource_lifecycle_dispatcher

    def close_lifecycle():
        order.append("lifecycle-dispatcher")
        outcome = lifecycle_attempts.pop(0)
        if outcome:
            original_lifecycle_close()
        return outcome

    bus.close_resource_lifecycle_dispatcher = close_lifecycle
    lifecycle = SequenceResourceLifecycleController(
        SimpleNamespace(
            remove_application_event_filter=lambda: None,
            disconnect_barcode_inputs=lambda: None,
        ),
        SequenceResourceLifecycleModel(),
        lifecycle_bus=bus,
    )
    lifecycle._shutdown_prepared_generation = 6
    assert lifecycle.finalize_application_shutdown(6)
    bus.register_resource_lifecycle_recipient(
        "disconnect-domains",
        "owners",
        lambda _request: order.append("owners") or True,
    )
    assert lifecycle.complete_application_shutdown_delivery(6)
    assert lifecycle.complete_application_shutdown_after_ready_ack(6) is False
    assert lifecycle.complete_application_shutdown_after_ready_ack(6)
    assert order == [
        "owners",
        "dispatcher",
        "lifecycle-dispatcher",
        "lifecycle-dispatcher",
    ]


def test_destroyed_shutdown_coordinator_rejects_late_completion_without_cleanup():
    bus = SequenceEventBus()
    model = SequenceWorkflowModel()
    model.shutdown_generation = 44
    model.phase = WorkflowPhase.SHUTDOWN_FLUSHING
    cleaned = []
    coordinator = SequenceShutdownCoordinator(
        model,
        bus,
        view=_ShutdownView(),
        cleanup_resources=lambda generation: cleaned.append(generation) or True,
    )
    coordinator.disconnect()
    assert coordinator.handle_shutdown_flush_completed(
        ShutdownFlushCompleted(44)
    ) is False
    assert cleaned == []


def test_main_close_source_has_no_polling_or_nested_modal_loop():
    from main_window import MainWindow

    source = inspect.getsource(MainWindow.closeEvent)
    assert "while True" not in source
    assert "processEvents" not in source
    assert ".exec_" not in source


def test_shutdown_formal_surface_has_no_third_close_handshake():
    from main_window import MainWindow
    from ui.sequence import sequence_messages
    from ui.sequence.sequence_widget import SequenceWindow

    assert not hasattr(sequence_messages, "ShutdownFinalized")
    assert not hasattr(sequence_messages, "ShutdownAcceptPermitted")
    assert not hasattr(MainWindow, "on_shutdown_accept_permitted")
    assert not hasattr(SequenceWindow, "register_shutdown_accept_recipient")
    assert not hasattr(SequenceWindow, "acknowledge_application_shutdown_finalized")


def test_main_hardware_guard_keeps_legacy_flags_only():
    from main_window import MainWindow

    source = inspect.getsource(MainWindow._sequence_audio_workflow_active)
    assert "player_status_flag" in source
    assert "_record_workflow_busy" in source
    assert "is_workflow_active" not in source


def test_native_qt_second_close_guard_exits_without_exception():
    script = r'''
import os
import sys
import types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
preferences = types.ModuleType("base.analysis_warning_preferences")
preferences.is_uncalibrated_microphone_warning_suppressed = lambda: False
preferences.save_uncalibrated_microphone_warning_suppressed = lambda *_args: None
sys.modules[preferences.__name__] = preferences
from PyQt5.QtWidgets import QApplication, QMainWindow
from main_window import MainWindow
from ui.sequence.sequence_messages import ShutdownReady

class Facade:
    def __init__(self):
        self.requests = []
    def is_workflow_active(self):
        return False
    def request_application_shutdown(self, generation):
        self.requests.append(generation)
        return True

class Window(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        MainWindow._initialize_application_shutdown_state(self)
        self.sequence_window = Facade()

app = QApplication.instance() or QApplication([])
window = Window()
window.show()
window.close()
assert window.sequence_window.requests == [0]
assert window.isVisible()
assert window.on_shutdown_ready(ShutdownReady(0)) is True
app.processEvents()
assert not window.isVisible()
'''
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_qt_ready_dispatch_contains_keyboardinterrupt_and_systemexit():
    script = r'''
import os
import sys
import types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
preferences = types.ModuleType("base.analysis_warning_preferences")
preferences.is_uncalibrated_microphone_warning_suppressed = lambda: False
preferences.save_uncalibrated_microphone_warning_suppressed = lambda *_args: None
sys.modules[preferences.__name__] = preferences
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QMainWindow
from main_window import MainWindow
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import ShutdownReady

class Window(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        MainWindow._initialize_application_shutdown_state(self)
        self._shutdown_active_generation = 5
        self.errors = [KeyboardInterrupt(), SystemExit(8)]
    def close(self):
        raise self.errors.pop(0)

app = QApplication.instance() or QApplication([])
window = Window()
bus = SequenceEventBus()
lifecycle = QObject()
bus.register_workflow_continuation_lifecycle_owner(lifecycle)
bus.register_workflow_continuation_recipient(
    "shutdown-ready", "main", window.on_shutdown_ready, owner=window
)
for _index in range(2):
    assert bus.deliver_workflow_continuation(
        ("shutdown-ready", 5),
        "shutdown-ready",
        ShutdownReady(5),
        owner=lifecycle,
    ) is False
assert window._shutdown_close_permission_generation == 5
'''
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
