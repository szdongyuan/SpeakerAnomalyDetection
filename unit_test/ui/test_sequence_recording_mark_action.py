from __future__ import annotations

import ast
import gc
import time
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from PyQt5 import sip
from PyQt5.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QObject,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget

from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    CancelWorkflowRequested,
    ConfigurationSnapshot,
    RecordingCancelled,
    RecordingMarkActionRequested,
    StartTestRequested,
)
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_service import (
    RecordingMarkActionRecoveryPending,
    RecordingMarkActionService,
)
from ui.sequence.sequence_recording_view import (
    SequenceRecordingAnalysisWindowsPort,
    SequenceRecordingMarkActionProjection,
    SequenceRecordingView,
)
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase


ROOT = Path(__file__).resolve().parents[2]
_QAPP = QApplication.instance() or QApplication([])


def SequenceEventBus(parent=None):
    """Build an explicitly authorized standalone Recording test harness."""
    return _SequenceEventBus(
        parent,
        standalone_recording_admission=True,
    )


class _Projection:
    def __init__(
        self,
        state,
        *,
        failure=None,
        restore_failure=None,
        finalize_failure=None,
    ):
        self.state = state
        self.failure = failure
        self.restore_failure = restore_failure
        self.finalize_failure = finalize_failure
        self.calls = []
        self.reenter = None

    def capture_mark_action_projection(self, _command):
        self.calls.append("capture")
        return dict(self.state)

    def apply_mark_action_projection(self, _command, _checkpoint):
        self.calls.append("apply")
        self.state["view"] = "cleared"
        if self.reenter is not None:
            self.calls.append(("nested", self.reenter()))
            self.reenter = None
        if self.failure is False:
            return False
        if isinstance(self.failure, BaseException):
            error = self.failure
            self.failure = None
            raise error
        return True

    def finalize_mark_action_projection(self, _command, _checkpoint):
        self.calls.append("finalize")
        if self.finalize_failure is False:
            return False
        if isinstance(self.finalize_failure, BaseException):
            error = self.finalize_failure
            self.finalize_failure = None
            raise error
        return True

    def restore_mark_action_projection(self, checkpoint, _error):
        self.calls.append("restore")
        if self.restore_failure is False:
            self.restore_failure = None
            return False
        if isinstance(self.restore_failure, BaseException):
            error = self.restore_failure
            self.restore_failure = None
            raise error
        self.state.clear()
        self.state.update(checkpoint)
        return True

    def fail_closed_mark_action_projection(self, _checkpoint, _error):
        self.calls.append("fail-closed")
        self.state["enabled"] = False
        return True


def _controller(projection, *, generation):
    model = RecordingModel()
    model.recorded_path = "record.wav"
    model.recorded_signal_info = {"file_path": "record.wav", "labels": "old"}
    statistics = object()
    model.statistics = statistics
    service = RecordingMarkActionService(
        model,
        projection,
        workflow_generation_provider=lambda: generation["value"],
    )
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        view=SequenceRecordingView(),
        mark_action_service=service,
        workflow_generation_provider=lambda: generation["value"],
        connect_queued=False,
    )
    return controller, service, model, statistics


def _wait_for_mark_dispatch(controller, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        calls = controller._mark_action_qt_owner_dispatch.snapshot()
        if calls:
            return calls[0]
        time.sleep(0.005)
    raise AssertionError("mark dispatch was not registered")


def test_mark_action_is_recording_owned_and_preserves_statistics_identity():
    state = {"view": "recorded", "enabled": True}
    projection = _Projection(state)
    generation = {"value": 7}
    controller, _service, model, statistics = _controller(
        projection, generation=generation
    )
    command = RecordingMarkActionRequested("mark-1", 7)

    assert controller.handle_mark_action(command) is True

    assert model.recorded_path is None
    assert model.recorded_signal_info == {}
    assert model.statistics is statistics
    assert state == {"view": "cleared", "enabled": True}
    assert projection.calls == ["capture", "apply", "finalize"]
    assert controller.handle_mark_action(command) is False


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_mark_action_rolls_back_partial_model_and_view_mutation(failure):
    state = {"view": "recorded", "enabled": True}
    projection = _Projection(state, failure=failure)
    generation = {"value": 4}
    controller, _service, model, statistics = _controller(
        projection, generation=generation
    )
    original_info = model.recorded_signal_info
    command = RecordingMarkActionRequested("mark-failure", 4)

    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        with pytest.raises(type(failure), match=str(failure)):
            controller.handle_mark_action(command)
    else:
        assert controller.handle_mark_action(command) is False

    assert model.recorded_path == "record.wav"
    assert model.recorded_signal_info is original_info
    assert model.statistics is statistics
    assert state == {"view": "recorded", "enabled": True}
    assert projection.calls == ["capture", "apply", "restore"]


def test_mark_action_incomplete_rollback_fails_closed_and_retries_exact_identity():
    state = {"view": "recorded", "enabled": True}
    projection = _Projection(
        state,
        failure=RuntimeError("forward"),
        restore_failure=KeyboardInterrupt("restore"),
    )
    generation = {"value": 9}
    controller, service, model, _statistics = _controller(
        projection, generation=generation
    )
    command = RecordingMarkActionRequested("mark-recovery", 9)
    other = RecordingMarkActionRequested("mark-other", 9)

    assert controller.handle_mark_action(command) is False
    assert isinstance(service.pending_recovery, RecordingMarkActionRecoveryPending)
    assert state["enabled"] is False
    assert controller.handle_mark_action(other) is False
    assert controller.handle_mark_action(command) is False
    assert projection.calls == ["capture", "apply", "restore", "fail-closed"]

    generation["value"] = 10
    quiesced = []
    controller.disconnect_quiesced.connect(quiesced.append)
    controller.disconnect()
    assert quiesced == []

    assert controller.request_mark_action() is False
    assert service.pending_recovery is None
    assert model.recorded_path == "record.wav"
    assert model.recorded_signal_info == {
        "file_path": "record.wav",
        "labels": "old",
    }
    assert quiesced == [""]
    assert projection.calls == [
        "capture",
        "apply",
        "restore",
        "fail-closed",
        "restore",
    ]


def test_pending_recovery_snapshot_is_frozen_and_caller_identity_cannot_poison_retry():
    projection = _Projection(
        {"view": "recorded", "enabled": True},
        failure=RuntimeError("forward"),
        restore_failure=False,
    )
    generation = {"value": 2}
    controller, service, _model, _statistics = _controller(
        projection, generation=generation
    )
    canonical = RecordingMarkActionRequested("canonical", 2)

    assert controller.handle_mark_action(canonical) is False
    snapshot = service.pending_recovery
    assert isinstance(snapshot, RecordingMarkActionRecoveryPending)
    assert snapshot.phase == "rollback"
    with pytest.raises(FrozenInstanceError):
        snapshot.command_id = "poison"
    assert controller.handle_mark_action(
        RecordingMarkActionRequested("canonical", 2)
    ) is False

    assert controller.request_mark_action() is False
    assert service.pending_recovery is None
    assert projection.calls[-1] == "restore"


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_finalize_failure_never_rolls_back_and_exact_retry_only_finalizes(failure):
    projection = _Projection(
        {"view": "recorded", "enabled": True},
        finalize_failure=failure,
    )
    generation = {"value": 12}
    controller, service, model, _statistics = _controller(
        projection, generation=generation
    )
    command = RecordingMarkActionRequested("finalize", 12)

    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        with pytest.raises(type(failure), match=str(failure)):
            controller.handle_mark_action(command)
    else:
        assert controller.handle_mark_action(command) is False

    assert model.recorded_path is None
    assert model.recorded_signal_info == {}
    assert service.pending_recovery is not None
    assert "restore" not in projection.calls

    projection.finalize_failure = None
    assert controller.request_mark_action() is True
    assert service.pending_recovery is None
    assert projection.calls == ["capture", "apply", "finalize", "finalize"]


def test_mark_action_reentry_stale_generation_change_and_shutdown_are_rejected():
    state = {"view": "recorded", "enabled": True}
    projection = _Projection(state)
    generation = {"value": 3}
    controller, _service, model, _statistics = _controller(
        projection, generation=generation
    )
    command = RecordingMarkActionRequested("mark-current", 3)
    projection.reenter = lambda: controller.handle_mark_action(command)
    assert controller.handle_mark_action(command) is True
    assert ("nested", False) in projection.calls

    stale = RecordingMarkActionRequested("mark-stale", 2)
    assert controller.handle_mark_action(stale) is False

    changing = RecordingMarkActionRequested("mark-changing", 3)
    projection.failure = None
    projection.reenter = lambda: generation.update(value=4)
    assert controller.handle_mark_action(changing) is False
    assert model.recorded_path is None  # restored to the pre-changing committed state
    assert state["view"] == "cleared"

    controller.disconnect()
    assert controller.handle_mark_action(
        RecordingMarkActionRequested("mark-shutdown", 4)
    ) is False


def test_mark_action_shutdown_during_projection_rolls_back_before_return():
    state = {"view": "recorded", "enabled": True}
    projection = _Projection(state)
    generation = {"value": 5}
    controller, _service, model, _statistics = _controller(
        projection, generation=generation
    )
    original_info = model.recorded_signal_info
    controller.disconnect_quiesced.connect(
        lambda _session_id: projection.calls.append("quiesced")
    )
    projection.reenter = controller.disconnect

    assert controller.handle_mark_action(
        RecordingMarkActionRequested("mark-shutdown-race", 5)
    ) is False

    assert model.recorded_path == "record.wav"
    assert model.recorded_signal_info is original_info
    assert state == {"view": "recorded", "enabled": True}
    assert projection.calls == [
        "capture",
        "apply",
        ("nested", None),
        "restore",
        "quiesced",
    ]


def test_mark_action_concurrent_delivery_has_one_mutation_owner():
    entered = Event()
    release = Event()

    class BlockingProjection(_Projection):
        def apply_mark_action_projection(self, command, checkpoint):
            entered.set()
            assert release.wait(2)
            return super().apply_mark_action_projection(command, checkpoint)

    projection = BlockingProjection({"view": "recorded", "enabled": True})
    generation = {"value": 6}
    controller, _service, _model, _statistics = _controller(
        projection, generation=generation
    )
    first = RecordingMarkActionRequested("mark-thread-1", 6)
    second = RecordingMarkActionRequested("mark-thread-2", 6)
    results = []
    worker = Thread(target=lambda: results.append(controller.handle_mark_action(first)))

    worker.start()
    assert entered.wait(2)
    assert controller.handle_mark_action(second) is False
    release.set()
    worker.join(2)
    _QAPP.processEvents()

    assert not worker.is_alive()
    assert results == [True]
    assert projection.calls == ["capture", "apply", "finalize"]


def test_mark_action_projection_owns_ordered_runtime_cleanup_and_exact_restore():
    calls = []

    class Field:
        def __init__(self, name, enabled=True):
            self.name = name
            self.enabled = enabled

        def isEnabled(self):
            return self.enabled

        def setEnabled(self, value):
            calls.append((self.name, bool(value)))
            self.enabled = bool(value)

        def setDisabled(self, value):
            return self.setEnabled(not value)

    class Board:
        def __init__(self):
            self.mode = "test"
            self.mark_btn = Field("mark", True)
            self.test_btn = Field("test", False)
            self.stacked_widget = SimpleNamespace(
                currentIndex=lambda: 0,
                setCurrentIndex=lambda value: calls.append(("stack", value)),
            )

        def on_mark_btn_clicked(self):
            calls.append("select-mark")
            self.mode = "mark"
            self.mark_btn.setEnabled(False)
            self.test_btn.setEnabled(True)

    data_struct = SimpleNamespace(
        store_wave_data="mono",
        store_wave_data_multi="multi",
        wav_calibration_metadata={"factor": 1},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    runtime = SimpleNamespace(
        count_board=Board(),
        data_struct=data_struct,
        signal_info={"signal": "old"},
        player_btn=Field("player", False),
        replayer_btn=Field("replay", True),
        data_btn=Field("data", True),
    )
    projection = SequenceRecordingMarkActionProjection(
        runtime,
        clear_plot=lambda: calls.append("clear-plot"),
        analysis_windows_port=SimpleNamespace(
            capture_mark_action_windows=lambda: "analysis-checkpoint",
            prepare_mark_action_windows=lambda checkpoint: calls.append(
                ("prepare-analysis", checkpoint)
            ),
            restore_mark_action_windows=lambda checkpoint, error: calls.append(
                ("restore-analysis", checkpoint)
            ),
            finalize_mark_action_windows=lambda checkpoint: calls.append(
                ("finalize-analysis", checkpoint)
            ),
        ),
    )
    command = RecordingMarkActionRequested("mark-view", 1)
    checkpoint = projection.capture_mark_action_projection(command)

    assert projection.apply_mark_action_projection(command, checkpoint) is True

    assert calls == [
        "select-mark",
        ("mark", False),
        ("test", True),
        "clear-plot",
        ("prepare-analysis", "analysis-checkpoint"),
        ("player", True),
        ("replay", False),
        ("data", False),
    ]
    assert data_struct.store_wave_data is None
    assert data_struct.store_wave_data_multi is None
    assert data_struct.wav_calibration_metadata is None
    assert data_struct.wav_calibration_metadata_authoritative is False
    assert data_struct.wav_calibration_warning_shown is False
    assert runtime.signal_info == {}
    assert runtime.player_btn.isEnabled() is True
    assert runtime.replayer_btn.isEnabled() is False
    assert runtime.data_btn.isEnabled() is False

    assert projection.restore_mark_action_projection(
        checkpoint, RuntimeError("rollback")
    )
    assert runtime.count_board.mode == "test"
    assert data_struct.store_wave_data == "mono"
    assert data_struct.store_wave_data_multi == "multi"
    assert data_struct.wav_calibration_metadata == {"factor": 1}
    assert runtime.signal_info == {"signal": "old"}
    assert runtime.player_btn.isEnabled() is False
    assert runtime.replayer_btn.isEnabled() is True
    assert runtime.data_btn.isEnabled() is True

    finalize_checkpoint = projection.capture_mark_action_projection(command)
    assert projection.apply_mark_action_projection(
        command, finalize_checkpoint
    ) is True
    assert projection.finalize_mark_action_projection(
        command, finalize_checkpoint
    ) is True
    assert calls[-1] == ("finalize-analysis", "analysis-checkpoint")


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_mark_action_order_rolls_back_every_failure_after_reversible_prepare(failure):
    calls = []

    class Field:
        def __init__(self, name, enabled, *, fail=None):
            self.name = name
            self.enabled = enabled
            self.fail = fail

        def isEnabled(self):
            return self.enabled

        def setEnabled(self, value):
            calls.append((self.name, bool(value)))
            self.enabled = bool(value)
            fail, self.fail = self.fail, None
            if fail is False:
                return False
            if isinstance(fail, BaseException):
                raise fail
            return None

    class Board:
        def __init__(self):
            self.mode = "test"
            self.mark_btn = Field("mark", True)
            self.test_btn = Field("test", False)
            self.stacked_widget = SimpleNamespace(
                currentIndex=lambda: 0,
                setCurrentIndex=lambda value: calls.append(("stack", value)),
            )

        def on_mark_btn_clicked(self):
            calls.append("select-mark")
            self.mode = "mark"
            self.mark_btn.setEnabled(False)
            self.test_btn.setEnabled(True)

    class AnalysisPort:
        prepared = False

        def capture_mark_action_windows(self):
            return "windows"

        def prepare_mark_action_windows(self, checkpoint):
            calls.append(("prepare-analysis", checkpoint))
            self.prepared = True

        def restore_mark_action_windows(self, checkpoint, _error):
            calls.append(("restore-analysis", checkpoint))
            self.prepared = False

        def finalize_mark_action_windows(self, checkpoint):
            calls.append(("finalize-analysis", checkpoint))

    data_struct = SimpleNamespace(
        store_wave_data="mono",
        store_wave_data_multi="multi",
        wav_calibration_metadata={"factor": 2},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    port = AnalysisPort()
    runtime = SimpleNamespace(
        count_board=Board(),
        data_struct=data_struct,
        signal_info={"signal": "old"},
        player_btn=Field("player", False),
        replayer_btn=Field("replay", True),
        data_btn=Field("data", True, fail=failure),
    )
    projection = SequenceRecordingMarkActionProjection(
        runtime,
        clear_plot=lambda: calls.append("clear-plot"),
        analysis_windows_port=port,
    )
    controller, service, model, _statistics = _controller(
        projection, generation={"value": 31}
    )
    original_info = model.recorded_signal_info
    command = RecordingMarkActionRequested("ordered-failure", 31)

    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        with pytest.raises(type(failure), match=str(failure)):
            controller.handle_mark_action(command)
    else:
        assert controller.handle_mark_action(command) is False

    assert calls[:8] == [
        "select-mark",
        ("mark", False),
        ("test", True),
        "clear-plot",
        ("prepare-analysis", "windows"),
        ("player", True),
        ("replay", False),
        ("data", False),
    ]
    assert ("restore-analysis", "windows") in calls
    assert ("finalize-analysis", "windows") not in calls
    assert service.pending_recovery is None
    assert model.recorded_path == "record.wav"
    assert model.recorded_signal_info is original_info
    assert runtime.count_board.mode == "test"
    assert data_struct.store_wave_data == "mono"
    assert data_struct.store_wave_data_multi == "multi"
    assert data_struct.wav_calibration_metadata == {"factor": 2}
    assert runtime.signal_info == {"signal": "old"}
    assert runtime.player_btn.isEnabled() is False
    assert runtime.replayer_btn.isEnabled() is True
    assert runtime.data_btn.isEnabled() is True
    assert port.prepared is False


class _CloseWindow:
    def __init__(self, name, *, close_failure=None, visible=True):
        self.name = name
        self.close_failure = close_failure
        self.visible = visible
        self.calls = []

    def isVisible(self):
        return self.visible

    def hide(self):
        self.calls.append("hide")
        self.visible = False

    def show(self):
        self.calls.append("show")
        self.visible = True

    def close(self):
        self.calls.append("close")
        if self.close_failure is False:
            return False
        if isinstance(self.close_failure, BaseException):
            raise self.close_failure
        return True


def _analysis_port(*windows):
    instances = list(windows)
    registry = {
        getattr(window, "name", None) or window.objectName(): window
        for window in windows
    }
    view = SimpleNamespace(
        model=SimpleNamespace(
            analysis_instances=instances,
            analysis_registry=registry,
        ),
        summary_window=None,
        feedback_dialogs=[],
        window_keys={},
    )
    return SequenceRecordingAnalysisWindowsPort(view), view, instances, registry


def test_analysis_windows_prepare_and_rollback_restore_exact_wrappers_and_visibility():
    first = _CloseWindow("first", visible=True)
    second = _CloseWindow("second", visible=False)
    port, view, instances, registry = _analysis_port(first, second)

    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    assert view.model.analysis_instances is instances
    assert view.model.analysis_registry is registry
    assert instances == [] and registry == {}
    assert first.visible is False and second.visible is False

    assert port.restore_mark_action_windows(
        checkpoint, RuntimeError("rollback")
    ) is True
    assert instances == [first, second]
    assert registry == {"first": first, "second": second}
    assert first.visible is True and second.visible is False


@pytest.mark.parametrize(
    "failure",
    [False, RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
)
def test_analysis_windows_finalize_preserves_exact_unclosed_registry(failure):
    closed = _CloseWindow("closed")
    failed = _CloseWindow("failed", close_failure=failure)
    waiting = _CloseWindow("waiting")
    port, view, instances, registry = _analysis_port(closed, failed, waiting)
    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True

    if isinstance(failure, BaseException):
        with pytest.raises(type(failure), match=str(failure)):
            port.finalize_mark_action_windows(checkpoint)
    else:
        assert port.finalize_mark_action_windows(checkpoint) is False

    assert closed.calls[-1] == "close"
    assert failed.calls[-1] == "close"
    assert "close" not in waiting.calls
    assert instances == [failed, waiting]
    assert registry == {"failed": failed, "waiting": waiting}
    assert port.restore_mark_action_windows(
        checkpoint, RuntimeError("too late")
    ) is False

    failed.close_failure = None
    assert port.finalize_mark_action_windows(checkpoint) is True
    assert instances == [] and registry == {}


def test_analysis_windows_finalize_handles_real_delete_on_close_deferred_delete():
    window = QWidget()
    window.setObjectName("mark-delete-on-close")
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    window.show()
    _QAPP.processEvents()
    port, view, instances, registry = _analysis_port(window)
    registry.clear()
    registry["real"] = window

    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    assert port.finalize_mark_action_windows(checkpoint) is True
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    _QAPP.processEvents()

    assert instances == []
    assert registry == {}
    with pytest.raises(RuntimeError):
        window.objectName()


def test_analysis_windows_finalize_does_not_drop_registry_only_window():
    orphan = _CloseWindow("orphan")
    view = SimpleNamespace(
        model=SimpleNamespace(
            analysis_instances=[],
            analysis_registry={"orphan": orphan},
        ),
        summary_window=None,
        feedback_dialogs=[],
        window_keys={},
    )
    port = SequenceRecordingAnalysisWindowsPort(view)

    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    assert port.finalize_mark_action_windows(checkpoint) is True

    assert orphan.calls == ["hide", "close"]
    assert view.model.analysis_registry == {}


def test_pending_recovery_reentry_concurrency_and_quiescence_are_terminal_once():
    entered = Event()
    release = Event()

    class BlockingRecoveryProjection(_Projection):
        block_recovery = False

        def restore_mark_action_projection(self, checkpoint, error):
            if self.block_recovery:
                entered.set()
                assert release.wait(2)
            return super().restore_mark_action_projection(checkpoint, error)

    projection = BlockingRecoveryProjection(
        {"view": "recorded", "enabled": True},
        failure=RuntimeError("forward"),
        restore_failure=False,
    )
    generation = {"value": 20}
    controller, service, _model, _statistics = _controller(
        projection, generation=generation
    )
    assert controller.handle_mark_action(
        RecordingMarkActionRequested("recovery-concurrent", 20)
    ) is False
    assert service.pending_recovery is not None
    quiesced = []
    controller.disconnect_quiesced.connect(quiesced.append)
    controller.disconnect()
    assert quiesced == []

    generation["value"] = 21
    controller.workflow_generation_provider = lambda: (_ for _ in ()).throw(
        AssertionError("forward generation gate must not run during compensation")
    )
    projection.block_recovery = True
    results = []
    worker = Thread(target=lambda: results.append(controller.request_mark_action()))
    worker.start()
    assert entered.wait(2)
    assert controller.request_mark_action() is False
    release.set()
    worker.join(2)
    _QAPP.processEvents()

    assert not worker.is_alive()
    assert results == [False]
    assert service.pending_recovery is None
    assert quiesced == [""]
    assert controller.request_mark_action() is False
    assert quiesced == [""]


def test_analysis_windows_native_deleted_before_capture_is_already_finalized():
    dead = QWidget()
    window_keys = weakref.WeakKeyDictionary()
    window_keys[dead] = "dead"
    dead.setAttribute(Qt.WA_DeleteOnClose, True)
    dead.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(dead)
    instances = [dead]
    registry = {"dead": dead}
    view = SimpleNamespace(
        model=SimpleNamespace(
            analysis_instances=instances,
            analysis_registry=registry,
        ),
        summary_window=dead,
        feedback_dialogs=[dead],
        window_keys=window_keys,
    )
    port = SequenceRecordingAnalysisWindowsPort(view)

    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    assert port.finalize_mark_action_windows(checkpoint) is True

    assert instances == []
    assert registry == {}
    assert view.summary_window is None
    assert view.feedback_dialogs == []
    assert list(window_keys.items()) == []


def test_analysis_windows_native_deleted_between_prepare_and_finalize_is_complete():
    window = QWidget()
    port, _view, instances, registry = _analysis_port(window)
    registry.clear()
    registry["window"] = window
    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(window)

    assert port.finalize_mark_action_windows(checkpoint) is True
    assert instances == []
    assert registry == {}


def test_analysis_windows_dead_between_retry_and_new_generation_entries_are_merged():
    class RetryNative(QWidget):
        name = "old"

        def close(self):
            return False

    old = RetryNative()
    port, view, instances, registry = _analysis_port(old)
    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    newer = _CloseWindow("newer")
    instances.append(newer)
    registry["newer"] = newer

    assert port.finalize_mark_action_windows(checkpoint) is False
    assert instances == [newer, old]
    assert registry == {"newer": newer, "old": old}

    # A native-deleted owned wrapper is confirmed complete on exact retry.
    old.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(old)

    assert port.finalize_mark_action_windows(checkpoint) is True
    assert instances == [newer]
    assert registry == {"newer": newer}


def test_analysis_windows_rollback_merges_new_generation_without_clearing_it():
    old = _CloseWindow("old")
    port, view, instances, registry = _analysis_port(old)
    checkpoint = port.capture_mark_action_windows()
    assert port.prepare_mark_action_windows(checkpoint) is True
    newer = _CloseWindow("newer")
    instances.append(newer)
    registry["old"] = newer
    registry["newer"] = newer
    view.summary_window = newer
    view.feedback_dialogs.append(newer)
    view.window_keys[newer] = "newer-key"

    assert port.restore_mark_action_windows(
        checkpoint, RuntimeError("rollback")
    ) is True

    assert newer in instances and old in instances
    assert registry["old"] is newer
    assert registry["newer"] is newer
    assert view.summary_window is newer
    assert newer in view.feedback_dialogs
    assert view.window_keys[newer] == "newer-key"


def test_mark_action_model_rollback_uses_identity_cas_and_preserves_new_recording():
    model = RecordingModel()
    model.recorded_path = "old.wav"
    model.recorded_signal_info = {"file_path": "old.wav"}
    newer_info = {"file_path": "new.wav"}

    class ReentrantProjection(_Projection):
        def apply_mark_action_projection(self, command, checkpoint):
            super().apply_mark_action_projection(command, checkpoint)
            model.recorded_path = "new.wav"
            model.recorded_signal_info = newer_info
            raise RuntimeError("forward failure after newer identity")

    service = RecordingMarkActionService(
        model,
        ReentrantProjection({"view": "recorded"}),
        workflow_generation_provider=lambda: 41,
    )

    assert service.apply(RecordingMarkActionRequested("model-cas", 41)) is False
    assert model.recorded_path == "new.wav"
    assert model.recorded_signal_info is newer_info
    assert service.pending_recovery is None


def test_mark_action_reservation_rejects_reentrant_begin_recording():
    projection = _Projection({"view": "recorded"})
    controller, _service, _model, _statistics = _controller(
        projection, generation={"value": 51}
    )
    preparations = []
    controller.prepare_session = lambda command: preparations.append(command)
    begin = BeginRecordingRequested(
        "begin-during-mark",
        "session-during-mark",
        False,
        {"workflow_generation": 51},
    )
    workflow_owner = QObject()
    workflow_capability = (
        controller.bus._bind_canonical_recording_workflow_owner(workflow_owner)
    )
    assert controller.bus._register_canonical_recording_admission(
        workflow_capability,
        begin,
    )
    projection.reenter = lambda: controller.handle_begin_recording(begin)

    assert controller.handle_mark_action(
        RecordingMarkActionRequested("mark-reservation", 51)
    ) is True
    assert preparations == []
    retry = BeginRecordingRequested(
        "begin-after-mark",
        "session-during-mark",
        False,
        {"workflow_generation": 51},
    )
    assert controller.bus._register_canonical_recording_admission(
        workflow_capability,
        retry,
    )
    assert controller.handle_begin_recording(retry) is False
    assert preparations == [retry]


def test_worker_mark_action_marshals_every_qwidget_touch_to_owner_qt_thread():
    owner_thread_id = int(QThread.currentThreadId())
    touch_threads = []

    class TrackingButton(QPushButton):
        def setEnabled(self, enabled):
            touch_threads.append(int(QThread.currentThreadId()))
            return super().setEnabled(enabled)

    class TrackingWindow(QWidget):
        def hide(self):
            touch_threads.append(int(QThread.currentThreadId()))
            return super().hide()

        def close(self):
            touch_threads.append(int(QThread.currentThreadId()))
            return super().close()

    class Board:
        mode = "test"

        def __init__(self):
            self.mark_btn = TrackingButton()
            self.test_btn = TrackingButton()
            self.stacked_widget = SimpleNamespace(
                currentIndex=lambda: 0,
                setCurrentIndex=lambda _value: None,
            )

        def on_mark_btn_clicked(self):
            touch_threads.append(int(QThread.currentThreadId()))
            self.mode = "mark"
            self.mark_btn.setEnabled(False)
            self.test_btn.setEnabled(True)

    analysis_window = TrackingWindow()
    analysis_view = SimpleNamespace(
        model=SimpleNamespace(
            analysis_instances=[analysis_window],
            analysis_registry={"analysis": analysis_window},
        ),
        summary_window=None,
        feedback_dialogs=[],
        window_keys={},
    )
    runtime = SimpleNamespace(
        count_board=Board(),
        data_struct=SimpleNamespace(
            store_wave_data="mono",
            store_wave_data_multi="multi",
            wav_calibration_metadata=None,
            wav_calibration_metadata_authoritative=False,
            wav_calibration_warning_shown=False,
        ),
        signal_info={"signal": "old"},
        player_btn=TrackingButton(),
        replayer_btn=TrackingButton(),
        data_btn=TrackingButton(),
    )
    projection = SequenceRecordingMarkActionProjection(
        runtime,
        clear_plot=lambda: touch_threads.append(int(QThread.currentThreadId())),
        analysis_windows_port=SequenceRecordingAnalysisWindowsPort(analysis_view),
    )
    controller, _service, _model, _statistics = _controller(
        projection, generation={"value": 61}
    )

    def run_in_qthread(callback):
        class Worker(QObject):
            done = pyqtSignal()

            def run(self):
                try:
                    results.append(callback())
                finally:
                    self.done.emit()

        results = []
        thread = QThread()
        worker = Worker()
        worker.moveToThread(thread)
        loop = QEventLoop()
        thread.started.connect(worker.run)
        worker.done.connect(thread.quit)
        worker.done.connect(loop.quit)
        thread.start()
        QTimer.singleShot(2000, loop.quit)
        loop.exec()
        assert thread.wait(2000)
        return results

    results = run_in_qthread(
        lambda: controller.handle_mark_action(
            RecordingMarkActionRequested("worker-mark", 61)
        )
    )

    assert results == [True]
    assert touch_threads
    assert set(touch_threads) == {owner_thread_id}

    class RetryWindow(TrackingWindow):
        reject_once = True

        def close(self):
            touch_threads.append(int(QThread.currentThreadId()))
            if self.reject_once:
                self.reject_once = False
                return False
            return QWidget.close(self)

    retry_window = RetryWindow()
    analysis_view.model.analysis_instances.append(retry_window)
    analysis_view.model.analysis_registry["retry"] = retry_window
    assert controller.handle_mark_action(
        RecordingMarkActionRequested("worker-retry", 61)
    ) is False
    assert _service.pending_recovery is not None

    retry_results = run_in_qthread(controller.request_mark_action)

    assert retry_results == [True]
    assert _service.pending_recovery is None
    assert set(touch_threads) == {owner_thread_id}


def test_worker_mark_dispatch_is_cancelled_by_native_controller_deletion_before_delivery():
    projection = _Projection({"view": "recorded"})
    projection.requires_qt_owner_thread = True
    controller, _service, _model, _statistics = _controller(
        projection, generation={"value": 71}
    )
    results = []
    worker = Thread(target=lambda: results.append(controller.request_mark_action()))

    worker.start()
    call = _wait_for_mark_dispatch(controller)
    call_ref = weakref.ref(call)
    del call
    sip.delete(controller)
    worker.join(1)

    assert not worker.is_alive()
    assert results == [False]
    gc.collect()
    assert call_ref() is None


def test_worker_mark_dispatch_is_cancelled_when_owner_qthread_finishes_without_delivery():
    class NonEventOwnerThread(QThread):
        def __init__(self):
            super().__init__()
            self.ready = Event()
            self.release = Event()
            self.controller = None

        def run(self):
            projection = _Projection({"view": "recorded"})
            projection.requires_qt_owner_thread = True
            self.controller, _service, _model, _statistics = _controller(
                projection, generation={"value": 72}
            )
            self.ready.set()
            assert self.release.wait(2)

    owner_thread = NonEventOwnerThread()
    owner_thread.start()
    assert owner_thread.ready.wait(2)
    controller = owner_thread.controller
    results = []
    worker = Thread(target=lambda: results.append(controller.request_mark_action()))

    worker.start()
    call = _wait_for_mark_dispatch(controller)
    call_ref = weakref.ref(call)
    del call
    owner_thread.release.set()
    assert owner_thread.wait(2000)
    worker.join(1)

    assert not worker.is_alive()
    assert results == [False]
    gc.collect()
    assert call_ref() is None
    sip.delete(controller)


def test_worker_mark_dispatch_wait_is_bounded_when_owner_loop_never_delivers():
    class StalledOwnerThread(QThread):
        def __init__(self):
            super().__init__()
            self.ready = Event()
            self.release = Event()
            self.controller = None

        def run(self):
            projection = _Projection({"view": "recorded"})
            projection.requires_qt_owner_thread = True
            self.controller, _service, _model, _statistics = _controller(
                projection, generation={"value": 74}
            )
            self.ready.set()
            assert self.release.wait(3)

    owner_thread = StalledOwnerThread()
    owner_thread.start()
    assert owner_thread.ready.wait(2)
    controller = owner_thread.controller
    started_at = time.monotonic()
    result = []
    worker = Thread(target=lambda: result.append(controller.request_mark_action()))

    worker.start()
    worker.join(1.5)
    elapsed = time.monotonic() - started_at

    assert not worker.is_alive()
    assert result == [False]
    assert elapsed < 1.5
    assert controller._mark_action_qt_owner_dispatch.snapshot() == ()

    owner_thread.release.set()
    assert owner_thread.wait(2000)
    sip.delete(controller)


@pytest.mark.parametrize(
    ("outcome", "expected_result", "expected_error", "expected_path"),
    [
        (True, True, None, None),
        (False, False, None, "record.wav"),
        (
            KeyboardInterrupt("claimed interruption"),
            None,
            KeyboardInterrupt,
            "record.wav",
        ),
        (SystemExit("claimed exit"), None, SystemExit, "record.wav"),
    ],
)
def test_claimed_mark_dispatch_waits_for_exact_terminal_outcome(
    outcome, expected_result, expected_error, expected_path
):
    entered = Event()
    release = Event()
    observer_done = Event()
    worker_completed_bounded = []
    pending_was_unfinished = []
    duplicate_results = []

    class BarrierProjection(_Projection):
        def apply_mark_action_projection(self, command, checkpoint):
            entered.set()
            assert release.wait(3)
            if outcome is False:
                self.failure = False
            elif isinstance(outcome, BaseException):
                self.failure = outcome
            return super().apply_mark_action_projection(command, checkpoint)

    projection = BarrierProjection({"view": "recorded"})
    projection.requires_qt_owner_thread = True
    controller, _service, model, _statistics = _controller(
        projection, generation={"value": 75}
    )
    results = []
    errors = []

    def invoke():
        try:
            results.append(controller.request_mark_action())
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=invoke)

    def observe_after_pending_timeout():
        assert entered.wait(2)
        time.sleep(1.1)
        worker_completed_bounded.append(not worker.is_alive())
        assert len(results) == 1
        pending_was_unfinished.append(results[0].terminal(0.0) is None)
        duplicate_results.append(controller.request_mark_action())
        release.set()
        observer_done.set()

    observer = Thread(target=observe_after_pending_timeout)
    worker.start()
    observer.start()
    _QAPP.processEvents()
    assert observer_done.wait(2)
    observer.join(1)
    worker.join(1)

    assert worker_completed_bounded == [True]
    assert pending_was_unfinished == [True]
    assert duplicate_results == [False]
    assert not worker.is_alive()
    assert len(results) == 1
    assert errors == []
    pending = results[0]
    assert type(pending) is not bool
    with pytest.raises(TypeError, match="pending"):
        bool(pending)
    terminal = pending.terminal(1.0)
    assert terminal is not None
    if expected_error is None:
        assert terminal.unwrap() is expected_result
    else:
        with pytest.raises(expected_error) as caught:
            terminal.unwrap()
        assert caught.value is outcome
        traceback_names = []
        current_traceback = caught.value.__traceback__
        while current_traceback is not None:
            traceback_names.append(current_traceback.tb_frame.f_code.co_name)
            current_traceback = current_traceback.tb_next
        assert "apply_mark_action_projection" in traceback_names
        assert "handle_mark_action" in traceback_names
        assert "_execute_mark_action_qt_owner_call" in traceback_names
    assert model.recorded_path == expected_path
    assert projection.state["view"] == (
        "cleared" if outcome is True else "recorded"
    )
    assert projection.calls == (
        ["capture", "apply", "finalize"]
        if outcome is True
        else ["capture", "apply", "restore"]
    )
    assert controller._mark_action_qt_owner_dispatch.snapshot() == ()
    call_ref = weakref.ref(pending._call)
    del pending
    results.clear()
    gc.collect()
    assert call_ref() is None


@pytest.mark.parametrize(
    "originating_error",
    [
        ValueError("fast failure"),
        KeyboardInterrupt("fast interruption"),
        SystemExit("fast exit"),
    ],
)
def test_fast_claimed_mark_dispatch_preserves_exception_provenance_and_gc(
    originating_error,
):
    projection = _Projection({"view": "recorded"})
    projection.requires_qt_owner_thread = True
    controller, service, _model, _statistics = _controller(
        projection, generation={"value": 76}
    )

    def originating_service_failure(*_args, **_kwargs):
        raise originating_error

    service.apply = originating_service_failure
    errors = []

    def invoke():
        try:
            controller.request_mark_action()
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=invoke)
    worker.start()
    call = _wait_for_mark_dispatch(controller)
    call_ref = weakref.ref(call)
    del call
    _QAPP.processEvents()
    worker.join(1)

    assert not worker.is_alive()
    assert errors == [originating_error]
    traceback_names = []
    current_traceback = errors[0].__traceback__
    while current_traceback is not None:
        traceback_names.append(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert "originating_service_failure" in traceback_names
    assert "handle_mark_action" in traceback_names
    assert "_execute_mark_action_qt_owner_call" in traceback_names
    assert controller._mark_action_qt_owner_dispatch.snapshot() == ()
    gc.collect()
    assert call_ref() is None


def test_pending_ordinary_dispatch_error_preserves_service_provenance_and_gc():
    entered = Event()
    release = Event()
    originating_error = ValueError("pending ordinary failure")
    projection = _Projection({"view": "recorded"})
    projection.requires_qt_owner_thread = True
    controller, service, _model, _statistics = _controller(
        projection, generation={"value": 77}
    )

    def originating_service_failure(*_args, **_kwargs):
        entered.set()
        assert release.wait(3)
        raise originating_error

    service.apply = originating_service_failure
    results = []
    worker = Thread(
        target=lambda: results.append(controller.request_mark_action())
    )
    worker.start()

    def release_after_bounded_handoff():
        assert entered.wait(2)
        time.sleep(1.1)
        assert not worker.is_alive()
        release.set()

    observer = Thread(target=release_after_bounded_handoff)
    observer.start()
    _QAPP.processEvents()
    observer.join(1)
    worker.join(1)

    assert len(results) == 1
    pending = results[0]
    terminal = pending.terminal(1.0)
    assert terminal is not None
    with pytest.raises(ValueError) as caught:
        terminal.unwrap()
    assert caught.value is originating_error
    traceback_names = []
    current_traceback = caught.value.__traceback__
    while current_traceback is not None:
        traceback_names.append(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert "originating_service_failure" in traceback_names
    assert "handle_mark_action" in traceback_names
    assert "_execute_mark_action_qt_owner_call" in traceback_names
    assert controller._mark_action_qt_owner_dispatch.snapshot() == ()
    call_ref = weakref.ref(pending._call)
    del pending
    results.clear()
    gc.collect()
    assert call_ref() is None


def test_mark_dispatch_cancel_is_idempotent_and_delivery_claim_wins_the_race():
    projection = _Projection({"view": "recorded"})
    projection.requires_qt_owner_thread = True
    controller, _service, _model, _statistics = _controller(
        projection, generation={"value": 73}
    )
    cancelled_results = []
    cancelled_worker = Thread(
        target=lambda: cancelled_results.append(controller.request_mark_action())
    )

    cancelled_worker.start()
    call = _wait_for_mark_dispatch(controller)
    call_ref = weakref.ref(call)
    del call
    controller._cancel_mark_action_qt_owner_calls()
    controller._cancel_mark_action_qt_owner_calls()
    cancelled_worker.join(1)
    _QAPP.processEvents()

    assert cancelled_results == [False]
    assert projection.calls == []
    gc.collect()
    assert call_ref() is None

    cancel_ran = Event()

    class ClaimedProjection(_Projection):
        def apply_mark_action_projection(self, command, checkpoint):
            def cancel_twice():
                controller._cancel_mark_action_qt_owner_calls()
                controller._cancel_mark_action_qt_owner_calls()
                cancel_ran.set()

            canceller = Thread(target=cancel_twice)
            canceller.start()
            assert cancel_ran.wait(1)
            canceller.join(1)
            return super().apply_mark_action_projection(command, checkpoint)

    claimed = ClaimedProjection({"view": "recorded"})
    claimed.requires_qt_owner_thread = True
    controller.mark_action_service.projection = claimed
    claimed_results = []
    claimed_worker = Thread(
        target=lambda: claimed_results.append(controller.request_mark_action())
    )

    claimed_worker.start()
    _wait_for_mark_dispatch(controller)
    _QAPP.processEvents()
    claimed_worker.join(1)

    assert not claimed_worker.is_alive()
    assert claimed_results == [True]
    assert claimed.calls == ["capture", "apply", "finalize"]
    assert controller._mark_action_qt_owner_dispatch.snapshot() == ()


def test_queued_workflow_start_blocked_by_mark_recovery_gets_one_formal_terminal():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "mark-blocked-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    projection = _Projection(
        {"view": "recorded"}, failure=False, restore_failure=False
    )
    recording_model = RecordingModel()
    recording_model.recorded_path = "record.wav"
    recording_model.recorded_signal_info = {"file_path": "record.wav"}
    service = RecordingMarkActionService(
        recording_model,
        projection,
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
    )
    prepared = []
    recording = SequenceRecordingController(
        recording_model,
        bus,
        view=SequenceRecordingView(),
        mark_action_service=service,
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    failures = []
    beginnings = []
    bus.events.recording_failed.connect(failures.append)
    bus.commands.begin_recording_requested.connect(beginnings.append)

    assert recording.handle_mark_action(
        RecordingMarkActionRequested("pending-mark", 0)
    ) is False
    assert service.pending_recovery is not None
    start = StartTestRequested("queued-start", "manual", "SN", False, 3)
    bus.commands.begin_recording_requested.emit(
        BeginRecordingRequested(
            "stale-hostile",
            "mark-blocked-session",
            False,
            {"workflow_generation": 0},
        )
    )
    bus.commands.begin_recording_requested.emit(
        BeginRecordingRequested(
            "collision-hostile",
            "mark-blocked-session",
            False,
            {"workflow_generation": 1},
        )
    )
    bus.commands.start_test_requested.emit(start)
    for _index in range(5):
        _QAPP.processEvents()

    assert workflow_model.phase is WorkflowPhase.IDLE
    assert len(failures) == 1
    assert failures[0].session_id == "mark-blocked-session"
    assert prepared == []
    assert recording.recent_identity_count == 0

    admitted = next(
        command for command in beginnings if command.command_id == start.command_id
    )
    bus.commands.begin_recording_requested.emit(admitted)
    bus.commands.begin_recording_requested.emit(admitted)
    for _index in range(3):
        _QAPP.processEvents()
    assert len(failures) == 1
    assert recording.recent_identity_count == 0

    assert recording.request_mark_action() is False
    assert service.pending_recovery is None
    bus.commands.start_test_requested.emit(start)
    for _index in range(5):
        _QAPP.processEvents()
    assert prepared == [beginnings[-1]]
    assert beginnings[-1].command_id == admitted.command_id
    assert workflow_model.phase is WorkflowPhase.IDLE
    assert recording.recent_identity_count == 1

    recording.disconnect()
    workflow.disconnect()


def test_replacement_recording_consumer_replays_live_workflow_admission():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "replacement-consumer-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    first = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda _command: None,
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    admitted = []
    failures = []
    bus.commands.begin_recording_requested.connect(admitted.append)
    bus.events.recording_failed.connect(failures.append)

    assert workflow.handle_start(
        StartTestRequested("replacement-consumer-start", "manual", "SN", False, 3)
    )
    assert workflow_model.phase is WorkflowPhase.PREPARING
    first.disconnect()

    prepared = []
    replacement = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    for _index in range(5):
        _QAPP.processEvents()

    assert len(admitted) == 2
    assert admitted[1] is admitted[0]
    assert prepared == [admitted[0]]
    assert len(failures) == 1
    assert failures[0].session_id == "replacement-consumer-session"
    assert workflow_model.phase is WorkflowPhase.IDLE

    replacement.disconnect()
    workflow.disconnect()


def test_replacement_after_consumer_gap_replays_cancellation_before_begin():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "cancelled-gap-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    admitted = []
    cancelled = []
    bus.commands.begin_recording_requested.connect(admitted.append)
    bus.events.recording_cancelled.connect(cancelled.append)

    assert workflow.handle_start(
        StartTestRequested("cancelled-gap-start", "manual", "SN", False, 3)
    )
    assert workflow.handle_cancel_workflow(
        CancelWorkflowRequested(
            "cancelled-gap-cancel",
            workflow_model.workflow_generation,
            "shutdown",
        )
    )
    assert workflow_model.phase is WorkflowPhase.CANCELLING

    prepared = []
    replacement = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    for _index in range(5):
        _QAPP.processEvents()

    assert len(admitted) == 2
    assert admitted[1] is admitted[0]
    assert prepared == []
    assert cancelled == [
        RecordingCancelled("cancelled-gap-session", "shutdown")
    ]
    assert workflow_model.phase is WorkflowPhase.IDLE

    replacement.disconnect()
    workflow.disconnect()


@pytest.mark.parametrize("release_kind", ["disconnect", "native-delete"])
def test_queued_workflow_begin_is_revoked_before_recording_delivery(
    release_kind,
):
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "revoked-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    prepared = []
    terminals = []
    recording = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    bus.events.recording_started.connect(terminals.append)
    bus.events.recording_completed.connect(terminals.append)
    bus.events.recording_failed.connect(terminals.append)
    bus.events.recording_cancelled.connect(terminals.append)

    assert workflow.handle_start(
        StartTestRequested("revoked-start", "manual", "SN", False, 3)
    )
    assert workflow_model.phase is WorkflowPhase.PREPARING
    if release_kind == "disconnect":
        workflow.disconnect()
    else:
        sip.delete(workflow)
    for _index in range(8):
        _QAPP.processEvents()

    assert prepared == []
    assert recording.recent_identity_count == 0
    assert terminals == []
    assert recording.model.active_session_id is None

    recording.disconnect()


@pytest.mark.parametrize("owner_lifecycle", ["live", "native-delete", "gc"])
def test_first_workflow_bind_revokes_claimed_standalone_before_retire(
    owner_lifecycle,
):
    bus = SequenceEventBus()
    prepared = []
    terminals = []
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    bus.events.recording_started.connect(terminals.append)
    bus.events.recording_completed.connect(terminals.append)
    bus.events.recording_failed.connect(terminals.append)
    bus.events.recording_cancelled.connect(terminals.append)
    retained_owners = []
    original_retire = bus._retire_canonical_recording_admission

    def bind_workflow_then_retire(capability, command):
        owner = QObject()
        assert bus._bind_canonical_recording_workflow_owner(owner) is not None
        if owner_lifecycle == "live":
            retained_owners.append(owner)
        elif owner_lifecycle == "native-delete":
            sip.delete(owner)
        else:
            owner_ref = weakref.ref(owner)
            del owner
            gc.collect()
            assert owner_ref() is None
        return original_retire(capability, command)

    bus._retire_canonical_recording_admission = bind_workflow_then_retire
    command = BeginRecordingRequested(
        "standalone-bind-race",
        "standalone-bind-session",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert prepared == []
    assert controller.recent_identity_count == 0
    assert terminals == []
    assert controller.model.active_session_id is None

    controller.disconnect()


def test_concurrent_first_workflow_bind_revokes_standalone_claim_before_retire():
    bus = SequenceEventBus()
    prepared = []
    terminals = []
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: prepared.append(command),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    bus.events.recording_started.connect(terminals.append)
    bus.events.recording_completed.connect(terminals.append)
    bus.events.recording_failed.connect(terminals.append)
    bus.events.recording_cancelled.connect(terminals.append)
    retire_entered = Event()
    release_retire = Event()
    original_retire = bus._retire_canonical_recording_admission

    def blocked_retire(capability, command):
        retire_entered.set()
        assert release_retire.wait(2)
        return original_retire(capability, command)

    bus._retire_canonical_recording_admission = blocked_retire
    command = BeginRecordingRequested(
        "standalone-concurrent-bind",
        "standalone-concurrent-session",
        False,
        {"workflow_generation": 7},
    )
    results = []
    worker = Thread(
        target=lambda: results.append(
            controller.handle_begin_recording(command)
        )
    )
    worker.start()
    assert retire_entered.wait(2)
    workflow_owner = QObject()
    assert bus._bind_canonical_recording_workflow_owner(workflow_owner) is not None
    release_retire.set()
    worker.join(2)

    assert not worker.is_alive()
    assert results == [False]
    assert prepared == []
    assert controller.recent_identity_count == 0
    assert terminals == []
    assert controller.model.active_session_id is None

    controller.disconnect()


def test_two_live_recording_consumers_use_one_bounded_standby_and_one_claim():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "standby-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    primary_prepared = []
    standby_prepared = []
    primary = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: primary_prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    standby = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: standby_prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    overflow = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda _command: pytest.fail("overflow prepared"),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    terminals = []
    bus.events.recording_failed.connect(terminals.append)

    assert primary._canonical_recording_admission_capability is not None
    assert standby._canonical_recording_admission_capability is not None
    assert overflow._canonical_recording_admission_capability is None
    assert workflow.handle_start(
        StartTestRequested("standby-start", "manual", "SN", False, 3)
    )
    for _index in range(5):
        _QAPP.processEvents()

    assert len(primary_prepared) == 1
    assert standby_prepared == []
    assert standby.recent_identity_count == 0
    assert overflow.recent_identity_count == 0
    assert len(terminals) == 1

    overflow.disconnect()
    standby.disconnect()
    primary.disconnect()
    workflow.disconnect()


def test_standby_promotion_replays_cancelled_admission_without_preparing():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "cancel-gap-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    primary = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda _command: pytest.fail("primary prepared"),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    standby_prepared = []
    standby = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: standby_prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    cancelled = []
    bus.events.recording_cancelled.connect(cancelled.append)

    assert workflow.handle_start(
        StartTestRequested("cancel-gap-start", "manual", "SN", False, 3)
    )
    assert workflow.handle_cancel_workflow(
        CancelWorkflowRequested(
            "cancel-gap-command",
            workflow_model.workflow_generation,
            "shutdown",
        )
    )
    assert workflow_model.phase is WorkflowPhase.CANCELLING
    primary.disconnect()
    for _index in range(5):
        _QAPP.processEvents()

    assert standby_prepared == []
    assert cancelled == [
        RecordingCancelled("cancel-gap-session", "shutdown")
    ]
    assert workflow_model.phase is WorkflowPhase.IDLE

    standby.disconnect()
    workflow.disconnect()


def test_begin_retirement_failure_rolls_back_identity_then_standby_replays():
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "retire-race-session",
        configuration_snapshot_provider=lambda: ConfigurationSnapshot(
            sequence_config={"mode": "RECORD_ONLY"},
            analysis_config={"auto_analysis": False},
            mic={"name": "input"},
            speaker=None,
            mic_channels=(0,),
        ),
    )
    primary_prepared = []
    standby_prepared = []
    primary = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: primary_prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    standby = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(),
        prepare_session=lambda command: standby_prepared.append(command),
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
        connect_queued=True,
    )
    admitted = []
    bus.commands.begin_recording_requested.connect(admitted.append)
    assert workflow.handle_start(
        StartTestRequested("retire-race-start", "manual", "SN", False, 3)
    )
    original_retire = bus._retire_canonical_recording_admission
    bus._retire_canonical_recording_admission = lambda *_args: False

    assert primary.handle_begin_recording(admitted[0]) is False
    assert primary_prepared == []
    assert primary.recent_identity_count == 0

    bus._retire_canonical_recording_admission = original_retire
    primary.disconnect()
    for _index in range(5):
        _QAPP.processEvents()

    assert standby_prepared == [admitted[0]]
    assert workflow_model.phase is WorkflowPhase.IDLE

    standby.disconnect()
    workflow.disconnect()


def test_sequence_window_mark_handler_is_a_one_step_recording_facade():
    source = (ROOT / "ui" / "sequence" / "sequence_widget.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    handler = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_mark_btn_clicked"
    )

    assert len(handler.body) == 1
    assert ast.unparse(handler.body[0]) == (
        "return self.recording_controller.request_mark_action()"
    )
    assert "def _clear_wav_calibration_runtime_state" not in source
    assert "analysis_windows_port=SequenceRecordingAnalysisWindowsPort" in source

    recording_view_source = (
        ROOT / "ui" / "sequence" / "sequence_recording_view.py"
    ).read_text(encoding="utf-8")
    recording_view_tree = ast.parse(recording_view_source)
    mark_projection = next(
        node
        for node in recording_view_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceRecordingMarkActionProjection"
    )
    assert "close_analysis_windows" not in ast.unparse(mark_projection)
