import ast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QObject, QThread, Qt, pyqtSignal

import ui.sequence.sequence_workflow_view as workflow_view_module
from ui.sequence.sequence_messages import ConfigurationSnapshot
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_messages import RecordingCompleted
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import (
    SessionOrigin,
    SequenceWorkflowModel,
    WorkflowPhase,
)
from ui.sequence.sequence_workflow_policy import (
    AutomaticAnalysisDecision,
    AutomaticAnalysisSource,
    SequenceAutomaticAnalysisPolicyService,
)
from ui.sequence.sequence_workflow_view import SequenceWorkflowView


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET = ROOT / "ui" / "sequence" / "sequence_widget.py"


def _configuration(*, auto_analysis=False):
    return ConfigurationSnapshot(
        sequence_config={"mode": "RECORD_ONLY"},
        analysis_config={"auto_analysis": auto_analysis},
    )


def _recording_snapshot(
    *,
    generation=4,
    mode="RECORD_ONLY",
    auto_analysis=True,
):
    return {
        "record_id": "record-1",
        "session": {
            "workflow_generation": generation,
            "configuration_generation": 2,
            "mode": mode,
            "analysis_config": {"auto_analysis": auto_analysis},
        },
    }


@pytest.mark.parametrize("mode", ["RECORD_ONLY", "PLAY_AND_RECORD"])
def test_recorded_policy_uses_frozen_session_flag_for_supported_modes(mode):
    service = SequenceAutomaticAnalysisPolicyService()
    live_configuration = _configuration(auto_analysis=False)

    decision = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=_recording_snapshot(mode=mode, auto_analysis=True),
        configuration_snapshot=live_configuration,
    )

    assert decision.source is AutomaticAnalysisSource.RECORDED
    assert decision.workflow_generation == 4
    assert decision.mode == mode
    assert decision.enabled is True


@pytest.mark.parametrize(
    "values",
    [
        (True, AutomaticAnalysisSource.RECORDED, None, False, "reason"),
        (1, "recorded", None, False, "reason"),
        (1, AutomaticAnalysisSource.RECORDED, 1, False, "reason"),
        (1, AutomaticAnalysisSource.RECORDED, None, 1, "reason"),
        (1, AutomaticAnalysisSource.RECORDED, None, False, object()),
    ],
)
def test_automatic_analysis_decision_rejects_coercible_fields(values):
    with pytest.raises((TypeError, ValueError)):
        AutomaticAnalysisDecision(*values)


@pytest.mark.parametrize(
    "value",
    [False, None, 0, 1, "", "true", (), object()],
)
def test_recorded_policy_does_not_coerce_hostile_auto_analysis_values(value):
    service = SequenceAutomaticAnalysisPolicyService()

    decision = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=_recording_snapshot(auto_analysis=value),
        configuration_snapshot=_configuration(auto_analysis=True),
    )

    assert decision.enabled is False


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"session": None},
        {"session": {}},
        {"session": {"workflow_generation": 4, "mode": "UNKNOWN"}},
        {
            "session": {
                "workflow_generation": 3,
                "mode": "RECORD_ONLY",
                "analysis_config": {"auto_analysis": True},
            }
        },
    ],
)
def test_recorded_policy_rejects_missing_malformed_or_stale_session(snapshot):
    service = SequenceAutomaticAnalysisPolicyService()

    decision = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=snapshot,
        configuration_snapshot=_configuration(),
    )

    assert decision.enabled is False


def test_import_policy_preserves_automatic_analysis_without_reading_auto_flag():
    service = SequenceAutomaticAnalysisPolicyService()

    decision = service.decide_imported(
        workflow_generation=9,
        recording_snapshot={"record_id": "imported-1"},
        configuration_snapshot=_configuration(auto_analysis=False),
    )

    assert decision.source is AutomaticAnalysisSource.IMPORTED
    assert decision.workflow_generation == 9
    assert decision.enabled is True


def test_policy_returns_one_immutable_decision_identity_per_workflow_generation():
    service = SequenceAutomaticAnalysisPolicyService()
    recording = _recording_snapshot()
    configuration = _configuration()

    def decide():
        return service.decide_recorded(
            workflow_generation=4,
            recording_snapshot=recording,
            configuration_snapshot=configuration,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        decisions = list(executor.map(lambda _index: decide(), range(64)))

    assert len({id(decision) for decision in decisions}) == 1
    with pytest.raises((AttributeError, TypeError)):
        decisions[0].enabled = False

    recording["session"]["analysis_config"]["auto_analysis"] = False
    repeated = decide()
    assert repeated is decisions[0]
    assert repeated.enabled is True

    cross_source_repeat = service.decide_imported(
        workflow_generation=4,
        recording_snapshot={"record_id": "imported-1"},
        configuration_snapshot=_configuration(),
    )
    assert cross_source_repeat is decisions[0]


def test_policy_decision_history_is_strictly_bounded():
    service = SequenceAutomaticAnalysisPolicyService()

    for generation in range(service.DECISION_HISTORY_LIMIT + 50):
        service.decide_recorded(
            workflow_generation=generation,
            recording_snapshot=_recording_snapshot(generation=generation),
            configuration_snapshot=_configuration(),
        )

    assert len(service._decisions) == service.DECISION_HISTORY_LIMIT


def test_policy_normalizes_hostile_mapping_baseexception_to_disabled_decision():
    class StopNow(BaseException):
        pass

    class Hostile(dict):
        def get(self, *_args, **_kwargs):
            raise StopNow("hostile mapping")

    service = SequenceAutomaticAnalysisPolicyService()
    decision = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=Hostile(),
        configuration_snapshot=_configuration(),
    )

    assert decision.enabled is False
    assert "snapshot" in decision.reason


def test_policy_reentry_cannot_publish_or_replace_the_generation_decision():
    class StopNow(BaseException):
        pass

    service = SequenceAutomaticAnalysisPolicyService()
    nested = []

    class Reentrant(dict):
        def get(self, *_args, **_kwargs):
            nested.append(
                service.decide_recorded(
                    workflow_generation=4,
                    recording_snapshot=_recording_snapshot(),
                    configuration_snapshot=_configuration(),
                )
            )
            raise StopNow("reentered")

    decision = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=Reentrant(),
        configuration_snapshot=_configuration(),
    )
    repeated = service.decide_recorded(
        workflow_generation=4,
        recording_snapshot=_recording_snapshot(),
        configuration_snapshot=_configuration(),
    )

    assert nested and nested[0].enabled is False
    assert "reentry" in nested[0].reason
    assert decision.enabled is False
    assert repeated is decision


def test_workflow_view_projects_awaiting_label_directly_from_canonical_model():
    model = SequenceWorkflowModel()
    refreshed = []
    synchronized = []

    def refresh():
        refreshed.append(True)

    def synchronize():
        synchronized.append(True)

    view = SequenceWorkflowView(
        model,
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    assert view.awaiting_label is False
    model.awaiting_label = True
    assert view.awaiting_label is True
    assert view.project_state_changed(object()) is True
    assert refreshed == [True]
    assert synchronized == [True]


def test_workflow_view_queued_delivery_is_dropped_when_qt_parent_is_deleted():
    script = r'''
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QObject, Qt, pyqtSignal

from ui.sequence.sequence_workflow_model import SequenceWorkflowModel
from ui.sequence.sequence_workflow_view import SequenceWorkflowView

class Sender(QObject):
    changed = pyqtSignal(object)

app = QCoreApplication.instance() or QCoreApplication([])
parent = QObject()
calls = []

def refresh():
    calls.append("refresh")

def synchronize():
    calls.append("shutdown")

view = SequenceWorkflowView(
    SequenceWorkflowModel(),
    refresh_player_button=refresh,
    synchronize_shutdown=synchronize,
    parent=parent,
)
sender = Sender()
sender.changed.connect(view.project_state_changed, Qt.QueuedConnection)
for index in range(128):
    sender.changed.emit(index)
parent.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert sip.isdeleted(parent)
assert sip.isdeleted(view)
app.processEvents()
assert calls == []

immediate_parent = QObject()
immediate_calls = []

def immediate_refresh():
    immediate_calls.append("refresh")

def immediate_synchronize():
    immediate_calls.append("shutdown")

immediate_view = SequenceWorkflowView(
    SequenceWorkflowModel(),
    refresh_player_button=immediate_refresh,
    synchronize_shutdown=immediate_synchronize,
    parent=immediate_parent,
)
sender.changed.connect(immediate_view.project_state_changed, Qt.QueuedConnection)
for index in range(128):
    sender.changed.emit(index)
sip.delete(immediate_parent)
assert sip.isdeleted(immediate_parent)
assert sip.isdeleted(immediate_view)
app.processEvents()
assert immediate_calls == []
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_workflow_view_does_not_retain_or_call_destroyed_callback_owner():
    class CallbackOwner(QObject):
        def refresh(self):
            raise AssertionError("destroyed refresh owner was called")

        def synchronize(self):
            raise AssertionError("destroyed shutdown owner was called")

    owner = CallbackOwner()
    owner_reference = weakref.ref(owner)
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=owner.refresh,
        synchronize_shutdown=owner.synchronize,
    )

    del owner
    gc.collect()

    assert owner_reference() is None
    assert view.project_state_changed() is False


def test_workflow_view_retains_ownerless_closure_only_for_view_lifetime():
    class CallbackOwner:
        pass

    owner = CallbackOwner()
    owner_reference = weakref.ref(owner)

    def refresh(callback_owner=owner):
        return callback_owner

    def synchronize(callback_owner=owner):
        return callback_owner

    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    del owner, refresh, synchronize
    gc.collect()

    assert owner_reference() is not None
    assert view.project_state_changed() is True

    del view
    gc.collect()

    assert owner_reference() is None


def test_workflow_view_retains_ownerless_inline_callbacks_for_its_lifetime():
    calls = []
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=lambda: calls.append("refresh"),
        synchronize_shutdown=lambda: calls.append("shutdown"),
    )

    gc.collect()

    assert view.project_state_changed() is True
    assert calls == ["refresh", "shutdown"]


def test_workflow_view_retains_inline_partials_until_view_release():
    calls = []

    def record(name):
        calls.append(name)

    refresh = partial(record, "refresh")
    synchronize = partial(record, "shutdown")
    refresh_reference = weakref.ref(refresh)
    synchronize_reference = weakref.ref(synchronize)
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    del refresh, synchronize
    gc.collect()

    assert refresh_reference() is not None
    assert synchronize_reference() is not None
    assert view.project_state_changed() is True
    assert calls == ["refresh", "shutdown"]

    del view
    gc.collect()

    assert refresh_reference() is None
    assert synchronize_reference() is None


def test_workflow_view_retains_callable_instances_until_view_release():
    calls = []

    class Callback:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            calls.append(self.name)

    refresh = Callback("refresh")
    synchronize = Callback("shutdown")
    refresh_reference = weakref.ref(refresh)
    synchronize_reference = weakref.ref(synchronize)
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    del refresh, synchronize
    gc.collect()

    assert refresh_reference() is not None
    assert synchronize_reference() is not None
    assert view.project_state_changed() is True
    assert calls == ["refresh", "shutdown"]

    del view
    gc.collect()

    assert refresh_reference() is None
    assert synchronize_reference() is None


def test_workflow_view_native_parent_deletion_releases_strong_callbacks():
    class Callback:
        def __call__(self):
            return None

    parent = QObject()
    refresh = Callback()
    synchronize = Callback()
    refresh_reference = weakref.ref(refresh)
    synchronize_reference = weakref.ref(synchronize)
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
        parent=parent,
    )
    del refresh, synchronize
    gc.collect()
    assert refresh_reference() is not None
    assert synchronize_reference() is not None

    sip.delete(parent)
    gc.collect()

    assert sip.isdeleted(view)
    assert refresh_reference() is None
    assert synchronize_reference() is None


def test_workflow_view_cyclic_gc_native_teardown_exits_cleanly():
    script = r'''
import gc
from PyQt5.QtWidgets import QApplication, QWidget

from ui.sequence.sequence_workflow_model import SequenceWorkflowModel
from ui.sequence.sequence_workflow_view import SequenceWorkflowView

app = QApplication.instance() or QApplication([])
for index in range(500):
    parent = QWidget()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=lambda: None,
        synchronize_shutdown=lambda: None,
        parent=parent,
    )
    parent.python_cycle = view
    view.python_cycle = parent
    del view, parent
    if index % 10 == 0:
        gc.collect()
gc.collect()
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_workflow_view_exact_disconnect_releases_strong_callbacks():
    class Sender(QObject):
        changed = pyqtSignal(object)

    class Callback:
        def __call__(self):
            return None

    sender = Sender()
    refresh = Callback()
    synchronize = Callback()
    refresh_reference = weakref.ref(refresh)
    synchronize_reference = weakref.ref(synchronize)
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )
    assert view.connect_state_changed(sender.changed) is True
    del refresh, synchronize
    gc.collect()
    assert refresh_reference() is not None
    assert synchronize_reference() is not None

    assert view.disconnect_state_changed(sender.changed) is True
    gc.collect()

    assert refresh_reference() is None
    assert synchronize_reference() is None
    assert view.project_state_changed() is False


def test_workflow_view_does_not_introspect_owner_fields_on_callable_instances():
    class HostileCallable:
        @property
        def __self__(self):
            raise SystemExit("must not inspect __self__")

        def __repr__(self):
            raise SystemExit("must not render callback")

        def __hash__(self):
            raise SystemExit("must not hash callback")

        def __eq__(self, _other):
            raise SystemExit("must not compare callback")

        def __call__(self):
            raise KeyboardInterrupt("callback stopped")

    refresh = HostileCallable()
    synchronize = HostileCallable()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    assert view.project_state_changed() is False


def test_workflow_view_contains_baseexception_from_callable_validation(monkeypatch):
    def hostile_callable(_value):
        raise KeyboardInterrupt("validation stopped")

    monkeypatch.setattr(
        workflow_view_module,
        "callable",
        hostile_callable,
        raising=False,
    )

    with pytest.raises(TypeError, match="ports must be callable"):
        SequenceWorkflowView(
            SequenceWorkflowModel(),
            refresh_player_button=lambda: None,
            synchronize_shutdown=lambda: None,
        )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_workflow_view_normalizes_callback_binding_baseexceptions(monkeypatch, error):
    class FailingReference:
        def __init__(self, *_args, **_kwargs):
            raise error

    monkeypatch.setattr(
        workflow_view_module,
        "_CallbackReference",
        FailingReference,
    )

    with pytest.raises(TypeError, match="callback binding failed"):
        SequenceWorkflowView(
            SequenceWorkflowModel(),
            refresh_player_button=lambda: None,
            synchronize_shutdown=lambda: None,
        )


def test_workflow_view_weakly_references_identifiable_builtin_bound_owner():
    owner = QObject()
    owner.setObjectName("callback-owner")
    owner_reference = weakref.ref(owner)
    synchronize = lambda: None
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=owner.objectName,
        synchronize_shutdown=synchronize,
    )

    del owner
    gc.collect()

    assert owner_reference() is None
    assert view.project_state_changed() is False


@pytest.mark.parametrize(
    "descriptor_error",
    [RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_workflow_view_builtin_bound_resolution_avoids_hostile_class_introspection(
    descriptor_error,
):
    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name in {"__mro__", "__dict__"}:
                raise SystemExit("class internals must not be inspected")
            return super().__getattribute__(name)

    class HostileList(list, metaclass=HostileMeta):
        fail_resolution = False

        def __getattribute__(self, name):
            if name == "append" and object.__getattribute__(
                self, "fail_resolution"
            ):
                raise descriptor_error
            return super().__getattribute__(name)

    owner = HostileList()
    callback = owner.append
    synchronize = lambda: None
    try:
        view = SequenceWorkflowView(
            SequenceWorkflowModel(),
            refresh_player_button=callback,
            synchronize_shutdown=synchronize,
        )
    except BaseException as error:
        raise AssertionError("constructor leaked hostile class behavior") from error
    owner.fail_resolution = True

    assert view.project_state_changed() is False


def test_workflow_view_connect_and_disconnect_are_strictly_thread_affine():
    script = r'''
from threading import Event

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QObject, QThread, pyqtSignal, pyqtSlot

from ui.sequence.sequence_workflow_model import SequenceWorkflowModel
from ui.sequence.sequence_workflow_view import SequenceWorkflowView

class Sender(QObject):
    changed = pyqtSignal(object)

class Commander(QObject):
    requested = pyqtSignal(object)

class CallbackOwner(QObject):
    def __init__(self):
        super().__init__()
        self.calls = []
    def refresh(self):
        self.calls.append("refresh")
    def synchronize(self):
        self.calls.append("shutdown")

class WorkerInvoker(QObject):
    def __init__(self, view, owner, sender_a):
        super().__init__()
        self.view = view
        self.owner = owner
        self.sender_a = sender_a
        self.results = {}
        self.completed = Event()
    @pyqtSlot(object)
    def run(self, action):
        if action == "connect":
            self.results[action] = self.view.connect_state_changed(
                self.sender_a.changed
            )
        elif action == "disconnect":
            self.results[action] = self.view.disconnect_state_changed(
                self.sender_a.changed
            )
        elif action == "teardown":
            sip.delete(self.view)
            sip.delete(self.owner)
            self.thread().quit()
        self.completed.set()

app = QCoreApplication.instance() or QCoreApplication([])
sender_a = Sender()
sender_b = Sender()
owner = CallbackOwner()
view = SequenceWorkflowView(
    SequenceWorkflowModel(),
    refresh_player_button=owner.refresh,
    synchronize_shutdown=owner.synchronize,
)
thread = QThread()
invoker = WorkerInvoker(view, owner, sender_a)
commander = Commander()
owner.moveToThread(thread)
view.moveToThread(thread)
invoker.moveToThread(thread)
commander.requested.connect(invoker.run)

assert view.connect_state_changed(sender_a.changed) is False
assert view._state_changed_signal is None
assert view._state_changed_receiver is None

thread.start()
commander.requested.emit("connect")
assert invoker.completed.wait(5)
invoker.completed.clear()
assert invoker.results["connect"] is True
current_receiver = view._state_changed_receiver

assert view.disconnect_state_changed(sender_a.changed) is False
assert view.connect_state_changed(sender_b.changed) is False
assert view._state_changed_receiver is current_receiver

sender_a.changed.emit("current")
for _ in range(100):
    if owner.calls:
        break
    QThread.msleep(5)
assert owner.calls == ["refresh", "shutdown"]

commander.requested.emit("disconnect")
assert invoker.completed.wait(5)
invoker.completed.clear()
assert invoker.results["disconnect"] is True
commander.requested.emit("teardown")
assert invoker.completed.wait(5)
assert thread.wait(5_000)
sip.delete(thread)
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "QObject:" not in result.stderr


def test_workflow_view_skips_callbacks_whose_qt_owner_was_immediately_deleted():
    class CallbackOwner(QObject):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def refresh(self):
            self.calls += 1

        def synchronize(self):
            self.calls += 1

    owner = CallbackOwner()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=owner.refresh,
        synchronize_shutdown=owner.synchronize,
    )

    sip.delete(owner)

    assert view.project_state_changed() is False
    assert owner.calls == 0


def test_workflow_view_rejects_qobject_callback_owner_from_another_thread():
    class CallbackOwner(QObject):
        def refresh(self):
            return None

        def synchronize(self):
            return None

    thread = QThread()
    owner = CallbackOwner()
    owner.moveToThread(thread)
    try:
        with pytest.raises(ValueError, match="thread"):
            SequenceWorkflowView(
                SequenceWorkflowModel(),
                refresh_player_button=owner.refresh,
                synchronize_shutdown=owner.synchronize,
            )
    finally:
        sip.delete(owner)
        sip.delete(thread)


def test_workflow_view_fails_closed_when_callback_owner_moves_or_thread_dies():
    class CallbackOwner(QObject):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def refresh(self):
            self.calls += 1

        def synchronize(self):
            self.calls += 1

    owner = CallbackOwner()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=owner.refresh,
        synchronize_shutdown=owner.synchronize,
    )
    thread = QThread()
    thread.start()
    owner.moveToThread(thread)
    try:
        assert view.project_state_changed() is False
        assert owner.calls == 0

        thread.quit()
        assert thread.wait(5_000) is True
        sip.delete(thread)

        assert view.project_state_changed() is False
        assert owner.calls == 0
    finally:
        if not sip.isdeleted(thread):
            thread.quit()
            thread.wait(5_000)
            sip.delete(thread)
        sip.delete(owner)


def test_workflow_view_connection_does_not_retain_deleted_parent_or_receiver():
    class Sender(QObject):
        changed = pyqtSignal(object)

    def refresh():
        return None

    def synchronize():
        return None

    sender = Sender()
    parent = QObject()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
        parent=parent,
    )
    parent_reference = weakref.ref(parent)
    view_reference = weakref.ref(view)
    assert view.connect_state_changed(sender.changed) is True

    sip.delete(parent)
    del view, parent
    gc.collect()

    assert parent_reference() is None
    assert view_reference() is None


@pytest.mark.parametrize(
    "error",
    [RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_workflow_view_contains_callback_baseexceptions_and_runs_both_ports(error):
    calls = []

    def refresh():
        calls.append("refresh")
        raise error

    def synchronize():
        calls.append("shutdown")

    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    assert view.project_state_changed() is False
    assert calls == ["refresh", "shutdown"]


def test_workflow_view_disconnect_is_exact_and_idempotent():
    class Sender(QObject):
        changed = pyqtSignal(object)

    calls = []

    def refresh():
        calls.append("refresh")

    def synchronize():
        calls.append("shutdown")

    sender = Sender()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )
    sender.changed.connect(lambda _message: calls.append("other"))

    assert view.connect_state_changed(sender.changed, Qt.DirectConnection) is True
    assert view.disconnect_state_changed(sender.changed) is True
    assert view.disconnect_state_changed(sender.changed) is False
    sender.changed.emit(object())

    assert calls == ["other"]


def test_workflow_view_disconnect_suppresses_already_queued_delivery():
    class Sender(QObject):
        changed = pyqtSignal(object)

    app = QCoreApplication.instance() or QCoreApplication([])
    calls = []

    def refresh():
        calls.append("refresh")

    def synchronize():
        calls.append("shutdown")

    sender = Sender()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )
    assert view.connect_state_changed(sender.changed, Qt.QueuedConnection) is True
    sender.changed.emit(object())

    assert view.disconnect_state_changed(sender.changed) is True
    app.processEvents()

    assert calls == []


def test_workflow_view_replacement_permanently_retires_old_queued_receiver():
    class Sender(QObject):
        changed = pyqtSignal(object)

    app = QCoreApplication.instance() or QCoreApplication([])
    calls = []

    def refresh():
        calls.append("refresh")

    def synchronize():
        calls.append("shutdown")

    sender_a = Sender()
    sender_b = Sender()
    view = SequenceWorkflowView(
        SequenceWorkflowModel(),
        refresh_player_button=refresh,
        synchronize_shutdown=synchronize,
    )

    assert view.connect_state_changed(sender_a.changed) is True
    sender_a.changed.emit("stale-a")
    assert view.connect_state_changed(sender_b.changed) is True
    sender_b.changed.emit("current-b")
    assert view.disconnect_state_changed(sender_a.changed) is False
    for cycle in range(20):
        current = sender_a if cycle % 2 else sender_b
        assert view.connect_state_changed(current.changed) is True
        current.changed.emit(cycle)
    assert view.connect_state_changed(sender_b.changed) is True
    sender_b.changed.emit("final-b")

    app.processEvents()

    assert calls == ["refresh", "shutdown"]


def test_workflow_controller_uses_policy_once_then_rejects_late_duplicate():
    model = SequenceWorkflowModel(workflow_generation=4)
    model.phase = WorkflowPhase.RECORDING
    model.active_session_id = "session-1"
    model.active_session_origin = SessionOrigin.CANONICAL
    model.configuration_snapshot = _configuration()
    bus = SequenceEventBus()
    analyses = []
    bus.commands.analysis_requested.connect(analyses.append)
    controller = SequenceWorkflowController(
        model,
        bus,
        analysis_id_factory=lambda: "analysis-1",
        connect_bus=False,
    )
    event = RecordingCompleted(
        "session-1",
        2,
        _recording_snapshot(generation=4, auto_analysis=True),
    )

    assert controller.handle_recording_completed(event) is True
    decision = model.automatic_analysis_decision
    assert decision is not None and decision.enabled is True
    assert model.awaiting_label is True
    assert model.phase is WorkflowPhase.ANALYZING
    assert len(analyses) == 1

    assert controller.handle_recording_completed(event) is False
    assert model.automatic_analysis_decision is decision
    assert len(analyses) == 1


@pytest.mark.parametrize("failure", [False, KeyboardInterrupt("policy stopped")])
def test_workflow_controller_restores_idle_when_policy_port_does_not_decide(failure):
    class FailingPolicy:
        def decide_recorded(self, **_kwargs):
            if failure is False:
                return False
            raise failure

        def decide_imported(self, **_kwargs):
            raise AssertionError("wrong policy source")

    model = SequenceWorkflowModel(workflow_generation=4)
    model.phase = WorkflowPhase.RECORDING
    model.active_session_id = "session-1"
    model.active_session_origin = SessionOrigin.CANONICAL
    model.configuration_snapshot = _configuration()
    controller = SequenceWorkflowController(
        model,
        SequenceEventBus(),
        automatic_analysis_policy=FailingPolicy(),
        connect_bus=False,
    )
    event = RecordingCompleted(
        "session-1",
        2,
        _recording_snapshot(generation=4, auto_analysis=True),
    )

    with pytest.raises((TypeError, KeyboardInterrupt)):
        controller.handle_recording_completed(event)

    assert model.phase is WorkflowPhase.IDLE
    assert model.awaiting_label is True
    assert model.automatic_analysis_decision is None


def test_sequence_window_has_no_policy_parser_or_duplicate_awaiting_label_state():
    source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in facade.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_workflow_automatic_analysis_policy" not in methods
    assert "SequenceAutomaticAnalysisPolicyService" in source
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_awaiting_ok_ng"
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
        for node in ast.walk(facade)
    )
    awaiting = methods["_awaiting_ok_ng"]
    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in awaiting.decorator_list
    )
    assert not any(
        isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
        for decorator in awaiting.decorator_list
    )
    awaiting_source = ast.get_source_segment(source, awaiting)
    assert "workflow_model.awaiting_label" in awaiting_source
    assert "workflow_view" not in awaiting_source
    view_construction = next(
        node
        for node in ast.walk(facade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SequenceWorkflowView"
    )
    assert any(
        keyword.arg == "parent"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "self"
        for keyword in view_construction.keywords
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect_state_changed"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "workflow_view"
        for node in ast.walk(facade)
    )
    project = methods["_project_workflow_state"]
    assert project.args.vararg is not None
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "project_state_changed"
        and any(isinstance(argument, ast.Starred) for argument in node.args)
        for node in ast.walk(project)
    )
