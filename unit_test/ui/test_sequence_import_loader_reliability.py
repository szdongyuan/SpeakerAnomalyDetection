from __future__ import annotations

import gc
import threading
import weakref
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QThread
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_recording_import_owner import SequenceRecordingImportController
from ui.sequence.sequence_recording_import_service import (
    AudioImportFailure,
    ImportedAudioStage,
    SequenceImportedAudioService,
)
from ui.sequence.sequence_recording_view import SequenceRecordingImportView
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_messages import (
    ConfigurationSnapshot,
    ImportedAudioFailed,
    ImportedAudioReady,
    LoadImportedAudioRequested,
)
from ui.sequence.sequence_event_bus import (
    ImportTerminalRecipientResult,
    SequenceEventBus,
)
from ui.sequence import sequence_messages as messages
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import (
    SequenceWorkflowModel,
    WorkflowPhase,
)
from ui.sequence.sequence_workflow_policy import (
    AutomaticAnalysisDecision,
    AutomaticAnalysisSource,
)


_QAPP = QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


def _configuration(mode="IMPORT_AUDIO"):
    return ConfigurationSnapshot(
        [{"seq1": {"acq": {"mode": mode, "detail": {}}}}], {}
    )


def _command(import_id="import-1", mode="IMPORT_AUDIO", path="new.wav"):
    return LoadImportedAudioRequested(
        "command-1", import_id, mode, path, _configuration(mode)
    )


def _stage(path="new.wav"):
    multi = np.zeros((3, 1), dtype=np.float32)
    return ImportedAudioStage(
        path,
        "IMPORT_AUDIO",
        32_000,
        multi,
        multi[:, 0].copy(),
        3,
        MappingProxyType({"recorded_channels": ()}),
    )


def _runtime():
    return SimpleNamespace(
        recorded_path="old.wav",
        recorded_signal_info={"file_path": "old.wav"},
        data_struct=SimpleNamespace(
            store_wave_data_multi=np.ones((2, 1), dtype=np.float32),
            store_wave_data=np.ones(2, dtype=np.float32),
            sample_rate=48_000,
            audio_lenth=2,
            stimulus_data=None,
            stimulus_info=None,
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=False,
        ),
    )


def _bus():
    return SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )


class _AlwaysAnalyzeImported:
    def decide_recorded(self, *, workflow_generation, **_kwargs):
        return AutomaticAnalysisDecision(
            workflow_generation,
            AutomaticAnalysisSource.RECORDED,
            None,
            False,
            "test policy",
        )

    def decide_imported(self, *, workflow_generation, **_kwargs):
        return AutomaticAnalysisDecision(
            workflow_generation,
            AutomaticAnalysisSource.IMPORTED,
            None,
            True,
            "test policy",
        )


def _workflow_import_bus(*, close_before_workflow=None):
    bus = SequenceEventBus()
    if close_before_workflow is not None:
        bus.register_import_terminal_recipient(
            "recording-close-before-workflow",
            close_before_workflow,
            critical=True,
        )
    workflow_model = SequenceWorkflowModel(configuration_generation=0)
    workflow_model.phase = WorkflowPhase.IMPORTING
    workflow_model.active_import_id = "import-1"
    workflow_model.configuration_snapshot = _configuration()
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        analysis_id_factory=lambda: "analysis-1",
        automatic_analysis_policy=_AlwaysAnalyzeImported(),
    )
    return bus, workflow_model, workflow


@pytest.mark.parametrize(
    "boundary",
    [
        "picker",
        "loader",
        "projection-capture",
        "projection-warning",
        "projection-show",
        "projection-enabled",
    ],
)
@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("closed callback failed"),
        lambda: KeyboardInterrupt("closed callback interrupted"),
        lambda: SystemExit("closed callback exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_reentrant_close_at_import_boundary_aborts_without_mutation_or_terminal(
    boundary, escape_factory, native_delete
):
    runtime = _runtime()
    old_runtime = {
        "recorded_path": runtime.recorded_path,
        "recorded_signal_info": runtime.recorded_signal_info,
        "multi": runtime.data_struct.store_wave_data_multi,
        "mono": runtime.data_struct.store_wave_data,
        "rate": runtime.data_struct.sample_rate,
        "length": runtime.data_struct.audio_lenth,
        "metadata": runtime.data_struct.wav_calibration_metadata,
    }
    bus = SequenceEventBus()
    raw_ready = []
    raw_failed = []
    bus.events.imported_audio_ready.connect(raw_ready.append)
    bus.events.imported_audio_failed.connect(raw_failed.append)
    holder = {}

    def close_then_escape():
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped

    def choose(selected):
        if boundary == "picker":
            close_then_escape()
        return selected

    def load(*_args):
        if boundary == "loader":
            close_then_escape()
        return _stage()

    def project(*_args):
        if boundary == "projection-show":
            close_then_escape()

    def capture_projection():
        if boundary == "projection-capture":
            close_then_escape()
        return ("old-plot", True)

    def set_enabled(_enabled):
        if boundary == "projection-enabled":
            close_then_escape()

    def present_warning(*_args):
        if boundary == "projection-warning":
            close_then_escape()

    view = SimpleNamespace(
        choose_import_audio_path=choose,
        capture_import_projection=capture_projection,
        restore_import_plot=lambda _state: True,
        restore_import_projection=lambda _state: True,
        present_import_warning=present_warning,
        clear_import_projection=lambda: None,
        show_imported_audio=project,
        set_import_data_enabled=set_enabled,
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=(
                (lambda *_args: ImportedAudioStage(
                    _stage().file_path,
                    _stage().mode,
                    _stage().sample_rate,
                    _stage().audio_multi,
                    _stage().audio_mono,
                    _stage().sample_count,
                    None,
                ))
                if boundary == "projection-warning"
                else load
            )
        ),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert raw_ready == []
    assert raw_failed == []
    assert bus.pending_import_terminal_delivery_count == 0
    assert runtime.recorded_path == old_runtime["recorded_path"]
    assert runtime.recorded_signal_info is old_runtime["recorded_signal_info"]
    assert runtime.data_struct.store_wave_data_multi is old_runtime["multi"]
    assert runtime.data_struct.store_wave_data is old_runtime["mono"]
    assert runtime.data_struct.sample_rate == old_runtime["rate"]
    assert runtime.data_struct.audio_lenth == old_runtime["length"]
    assert runtime.data_struct.wav_calibration_metadata is old_runtime["metadata"]
    assert model.active_import_id is None


@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("restore callback failed"),
        lambda: KeyboardInterrupt("restore callback interrupted"),
        lambda: SystemExit("restore callback exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_reentrant_close_during_projection_restore_stops_remaining_qt_callbacks(
    escape_factory, native_delete
):
    runtime = _runtime()
    old_runtime = {
        "recorded_path": runtime.recorded_path,
        "recorded_signal_info": runtime.recorded_signal_info,
        "multi": runtime.data_struct.store_wave_data_multi,
        "mono": runtime.data_struct.store_wave_data,
        "rate": runtime.data_struct.sample_rate,
        "length": runtime.data_struct.audio_lenth,
        "metadata": runtime.data_struct.wav_calibration_metadata,
    }
    bus = SequenceEventBus()
    holder = {}
    enabled_calls = []

    def restore_plot(_state):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped
        return True

    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=lambda: ("old-plot", True),
        restore_import_plot=restore_plot,
        present_import_warning=lambda *_args: None,
        show_imported_audio=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("projection failed")
        ),
        set_import_data_enabled=enabled_calls.append,
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert enabled_calls == []
    assert bus.pending_import_terminal_delivery_count == 0
    assert runtime.recorded_path == old_runtime["recorded_path"]
    assert runtime.recorded_signal_info is old_runtime["recorded_signal_info"]
    assert runtime.data_struct.store_wave_data_multi is old_runtime["multi"]
    assert runtime.data_struct.store_wave_data is old_runtime["mono"]
    assert runtime.data_struct.sample_rate == old_runtime["rate"]
    assert runtime.data_struct.audio_lenth == old_runtime["length"]
    assert runtime.data_struct.wav_calibration_metadata is old_runtime["metadata"]
    assert model.active_import_id is None


@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("terminal recipient failed"),
        lambda: KeyboardInterrupt("terminal recipient interrupted"),
        lambda: SystemExit("terminal recipient exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_reentrant_close_during_terminal_nack_abandons_exact_reservation(
    escape_factory, native_delete
):
    bus = SequenceEventBus()
    holder = {}

    def recipient(_event):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped
        return False

    bus.register_import_terminal_recipient("workflow", recipient, critical=True)
    raw_ready = []
    raw_failed = []
    bus.events.imported_audio_ready.connect(raw_ready.append)
    bus.events.imported_audio_failed.connect(raw_failed.append)
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    delivery_id = ("ImportedAudioReady", "import-1")
    assert raw_ready == []
    assert raw_failed == []
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.import_terminal_abandonment_reason(delivery_id) == "recording-disconnect"
    assert model.active_import_id == "import-1"


@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_canonical_ack_after_critical_close_keeps_committed_runtime_and_analysis(
    native_delete
):
    holder = {}

    def close_before_workflow(_event):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        return True

    bus, workflow_model, workflow = _workflow_import_bus(
        close_before_workflow=close_before_workflow
    )
    analyses = []
    raw_ready = []
    bus.commands.analysis_requested.connect(analyses.append)
    bus.events.imported_audio_ready.connect(raw_ready.append)
    runtime = _runtime()
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is True
    assert runtime.recorded_path == "new.wav"
    assert model.import_runtime_consistent is True
    assert model.active_import_id is None
    assert workflow_model.phase is WorkflowPhase.ANALYZING
    assert len(analyses) == 1
    assert analyses[0].recording_snapshot["file_path"] == "new.wav"
    assert raw_ready == []
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1
    assert bus.import_terminal_abandonment_reason(
        ("ImportedAudioReady", "import-1")
    ) is None
    assert workflow is not None


@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_raw_observer_close_after_canonical_ack_cannot_veto_commit(native_delete):
    bus, workflow_model, workflow = _workflow_import_bus()
    holder = {}
    raw_ready = []
    analyses = []
    bus.commands.analysis_requested.connect(analyses.append)

    def close_after_ack(event):
        raw_ready.append(event)
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True

    bus.events.imported_audio_ready.connect(close_after_ack)
    runtime = _runtime()
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is True
    assert runtime.recorded_path == "new.wav"
    assert model.import_runtime_consistent is True
    assert model.active_import_id is None
    assert workflow_model.phase is WorkflowPhase.ANALYZING
    assert len(analyses) == 1
    assert analyses[0].recording_snapshot["file_path"] == "new.wav"
    assert len(raw_ready) == 1
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1
    assert workflow is not None


@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_failed_terminal_ack_after_critical_close_retires_exact_import(native_delete):
    holder = {}

    def close_before_workflow(_event):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        return True

    bus, workflow_model, workflow = _workflow_import_bus(
        close_before_workflow=close_before_workflow
    )
    raw_failed = []
    bus.events.imported_audio_failed.connect(raw_failed.append)
    runtime = _runtime()
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            present_import_warning=lambda *_args: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda *_args: (_ for _ in ()).throw(
                AudioImportFailure(
                    "decode failed", "提示", "decode failed", clear_runtime=False
                )
            )
        ),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert runtime.recorded_path == "old.wav"
    assert model.active_import_id is None
    assert workflow_model.phase is WorkflowPhase.IDLE
    assert raw_failed == []
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1
    assert bus.import_terminal_abandonment_reason(
        ("ImportedAudioFailed", "import-1")
    ) is None
    assert workflow is not None


@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_reentrant_close_during_terminal_classification_abandons_reservation(
    native_delete
):
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    holder = {}

    def classify(_delivery_id, _event):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        return ImportTerminalRecipientResult.PERMANENT_REJECT

    bus.classify_import_terminal_delivery = classify
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    delivery_id = ("ImportedAudioReady", "import-1")
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.import_terminal_abandonment_reason(delivery_id) == "recording-disconnect"
    assert model.active_import_id == "import-1"


@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("terminal logger failed"),
        lambda: KeyboardInterrupt("terminal logger interrupted"),
        lambda: SystemExit("terminal logger exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_terminal_delivery_error_logger_close_abandons_created_reservation(
    escape_factory, native_delete
):
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    original_deliver = bus.deliver_import_terminal
    holder = {}

    def deliver_then_raise(delivery_id, event):
        assert original_deliver(delivery_id, event) is False
        raise RuntimeError("dispatcher escaped")

    def close_then_escape(*_args):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped

    bus.deliver_import_terminal = deliver_then_raise
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
        logger=SimpleNamespace(error=close_then_escape),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    delivery_id = ("ImportedAudioReady", "import-1")
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.import_terminal_abandonment_reason(delivery_id) == "recording-disconnect"
    assert model.active_import_id == "import-1"


@pytest.mark.parametrize(
    "first_escape",
    [None, RuntimeError("nack failed"), KeyboardInterrupt("nack interrupted"), SystemExit("nack exited")],
    ids=["nack", "ordinary", "keyboard-interrupt", "system-exit"],
)
def test_deleted_terminal_retry_timer_is_replaced_and_exact_reservation_recovers(
    first_escape
):
    bus = SequenceEventBus()
    accept = {"value": False}
    deliveries = []

    def recipient(event):
        deliveries.append(event)
        if not accept["value"] and first_escape is not None:
            raise first_escape
        return accept["value"]

    bus.register_import_terminal_recipient(
        "workflow", recipient, critical=True
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    timer = controller.import_terminal_retry_timer
    assert timer is not None
    sip.delete(timer)

    assert controller.handle_load_imported_audio_requested(_command()) is False
    delivery_id = ("ImportedAudioReady", "import-1")
    replacement = controller.import_terminal_retry_timer
    assert replacement is not None
    assert replacement is not timer
    assert replacement.thread() is controller.thread()
    assert replacement.isSingleShot() is True
    assert replacement.isActive() is True
    assert controller.pending_import_terminal_identity == "import-1"
    assert bus.pending_import_terminal_delivery_count == 1
    assert bus.import_terminal_abandonment_reason(delivery_id) is None
    assert model.active_import_id == "import-1"

    accept["value"] = True
    assert controller.retry_pending_import_terminal("import-1") is True
    assert controller.pending_import_terminal_identity is None
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1
    assert model.active_import_id is None
    assert len(deliveries) == 2


@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_retry_ack_after_critical_close_completes_without_abandon(native_delete):
    bus = SequenceEventBus()
    holder = {}
    attempts = []

    def recipient(event):
        attempts.append(event)
        if len(attempts) == 1:
            return False
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        return True

    bus.register_import_terminal_recipient("workflow", recipient, critical=True)
    runtime = _runtime()
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert controller.retry_pending_import_terminal("import-1") is True
    delivery_id = ("ImportedAudioReady", "import-1")
    assert runtime.recorded_path == "new.wav"
    assert model.active_import_id is None
    assert controller.pending_import_terminal_identity is None
    assert bus.pending_import_terminal_delivery_count == 0
    assert bus.completed_import_terminal_delivery_count == 1
    assert bus.import_terminal_abandonment_reason(delivery_id) is None
    assert len(attempts) == 2


def test_concurrent_deleted_timer_replacement_is_marshaled_once_to_owner_thread():
    bus = SequenceEventBus()
    accept = {"value": False}
    deliveries = []
    bus.register_import_terminal_recipient(
        "workflow",
        lambda event: deliveries.append(event) or accept["value"],
        critical=True,
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    assert controller.handle_load_imported_audio_requested(_command()) is False
    original = controller.import_terminal_retry_timer
    assert original is not None
    sip.delete(original)
    results = []
    workers = [
        threading.Thread(
            target=lambda: results.append(
                controller._schedule_import_terminal_retry()
            )
        )
        for _index in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.is_alive() is False

    assert results == [True] * 8
    _QAPP.processEvents()
    replacement = controller.import_terminal_retry_timer
    assert replacement is not None
    assert replacement is not original
    assert replacement.thread() == controller.thread()
    assert replacement.isActive() is True
    assert bus.pending_import_terminal_delivery_count == 1

    accept["value"] = True
    assert controller.retry_pending_import_terminal("import-1") is True
    reference = weakref.ref(controller)
    sip.delete(controller)
    del controller
    gc.collect()
    assert reference() is None
    assert len(deliveries) == 2


@pytest.mark.parametrize(
    "terminal_outcome",
    [
        True,
        False,
        RuntimeError("recipient failed"),
        KeyboardInterrupt("recipient interrupted"),
        SystemExit("recipient exited"),
    ],
    ids=["ack", "nack", "ordinary", "keyboard-interrupt", "system-exit"],
)
def test_blocked_terminal_concurrent_duplicate_and_disconnect_has_exclusive_history(
    terminal_outcome
):
    bus = SequenceEventBus()
    holder = {}
    attempts = []
    reentrant_results = []
    duplicate_results = []
    disconnect_results = []
    worker_errors = []

    def concurrent_call(callback, output):
        try:
            output.append(callback())
        except BaseException as error:
            worker_errors.append(error)

    def recipient(event):
        attempts.append(event)
        if len(attempts) == 1:
            return False
        controller = holder["controller"]
        reentrant_results.append(
            controller.retry_pending_import_terminal("import-1")
        )
        assert controller.import_terminal_retry_timer.isActive() is False
        duplicate = threading.Thread(
            target=concurrent_call,
            args=(
                lambda: controller.retry_pending_import_terminal("import-1"),
                duplicate_results,
            ),
        )
        closer = threading.Thread(
            target=concurrent_call,
            args=(controller.disconnect, disconnect_results),
        )
        duplicate.start()
        closer.start()
        duplicate.join(5)
        closer.join(5)
        assert duplicate.is_alive() is False
        assert closer.is_alive() is False
        if isinstance(terminal_outcome, BaseException):
            raise terminal_outcome
        return terminal_outcome

    bus.register_import_terminal_recipient("workflow", recipient, critical=True)
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    retry_result = controller.retry_pending_import_terminal("import-1")
    _QAPP.processEvents()
    delivery_id = ("ImportedAudioReady", "import-1")
    completed = bus.completed_import_terminal_delivery_count
    abandoned = bus.abandoned_import_terminal_delivery_count

    assert worker_errors == []
    assert reentrant_results == [False]
    assert duplicate_results == [False]
    assert disconnect_results == [True]
    assert len(attempts) == 2
    assert (completed, abandoned) in {(1, 0), (0, 1)}
    assert (completed == 1) is (terminal_outcome is True)
    assert retry_result is (terminal_outcome is True)
    assert bus.pending_import_terminal_delivery_count == 0
    assert controller.pending_import_terminal_identity is None
    if terminal_outcome is True:
        assert bus.import_terminal_abandonment_reason(delivery_id) is None
        assert model.active_import_id is None
    else:
        assert bus.import_terminal_abandonment_reason(delivery_id) == "recording-disconnect"
        assert model.active_import_id == "import-1"


def test_off_thread_load_cancel_and_retry_reject_before_external_work_or_timer_stop():
    bus = SequenceEventBus()
    deliveries = []
    bus.register_import_terminal_recipient(
        "workflow", lambda event: deliveries.append(event) or False, critical=True
    )
    view_calls = []
    load_calls = []
    provider_calls = []
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: view_calls.append(selected)
            or selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(
            load=lambda *_args: load_calls.append(True) or _stage()
        ),
    )

    results = []
    errors = []

    def invoke(callback):
        try:
            results.append(callback())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(
        target=invoke,
        args=(lambda: controller.handle_load_imported_audio_requested(_command()),),
    )
    worker.start()
    worker.join(5)
    assert worker.is_alive() is False
    assert errors == []
    assert results == [False]
    assert view_calls == []
    assert load_calls == []
    assert model.active_import_id is None

    assert controller.handle_load_imported_audio_requested(_command()) is False
    timer = controller.import_terminal_retry_timer
    assert timer is not None and timer.isActive() is True
    controller.workflow_identity_provider = (
        lambda: provider_calls.append(True)
        or {
            "import_id": "import-1",
            "phase": "CANCELLING",
            "workflow_generation": 0,
        }
    )
    delivery_count = len(deliveries)
    results.clear()
    workers = [
        threading.Thread(
            target=invoke,
            args=(
                lambda: controller.handle_cancel_imported_audio_requested(
                    messages.CancelImportedAudioRequested("import-1", 0, "cancel")
                ),
            ),
        ),
        threading.Thread(
            target=invoke,
            args=(lambda: controller.retry_pending_import_terminal("import-1"),),
        ),
    ]
    for current in workers:
        current.start()
    for current in workers:
        current.join(5)
        assert current.is_alive() is False

    assert errors == []
    assert sorted(results) == [False, False]
    assert provider_calls == []
    assert model.import_cancel_pending is False
    assert len(deliveries) == delivery_count
    assert timer.isActive() is True
    assert controller.pending_import_terminal_identity == "import-1"

    reference = weakref.ref(controller)
    sip.delete(controller)
    results.clear()
    deleted_worker = threading.Thread(
        target=invoke,
        args=(lambda: controller.handle_load_imported_audio_requested(_command()),),
    )
    deleted_worker.start()
    deleted_worker.join(5)
    assert deleted_worker.is_alive() is False
    assert errors == []
    assert results == [False]
    del controller
    gc.collect()
    assert reference() is None


def test_controller_with_exited_qthread_rejects_import_without_external_work_and_collects():
    view_calls = []
    load_calls = []
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: view_calls.append(selected)
            or selected
        ),
        bus=_bus(),
        runtime=_runtime(),
        import_service=SimpleNamespace(
            load=lambda *_args: load_calls.append(True)
            or (_ for _ in ()).throw(SystemExit("must not run"))
        ),
    )
    owner_thread = QThread()
    controller.moveToThread(owner_thread)
    owner_thread.start()

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert view_calls == []
    assert load_calls == []
    owner_thread.quit()
    assert owner_thread.wait(5_000) is True
    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert view_calls == []
    assert load_calls == []

    reference = weakref.ref(controller)
    sip.delete(controller)
    del controller
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("boundary", ["warning", "logger"])
@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("close boundary failed"),
        lambda: KeyboardInterrupt("close boundary interrupted"),
        lambda: SystemExit("close boundary exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_reentrant_close_from_warning_or_log_aborts_without_terminal(
    boundary, escape_factory, native_delete
):
    runtime = _runtime()
    old_multi = runtime.data_struct.store_wave_data_multi
    bus = SequenceEventBus()
    holder = {}

    def close_then_escape(*_args):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped

    logger = SimpleNamespace(
        error=close_then_escape if boundary == "logger" else lambda *_args: None
    )
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=(
            (lambda: (_ for _ in ()).throw(RuntimeError("capture failed")))
            if boundary == "logger"
            else lambda: None
        ),
        present_import_warning=(
            close_then_escape if boundary == "warning" else lambda *_args: None
        ),
        clear_import_projection=lambda: None,
        show_imported_audio=lambda *_args: None,
        set_import_data_enabled=lambda _enabled: None,
    )
    service = SimpleNamespace(
        load=(
            (lambda *_args: (_ for _ in ()).throw(
                AudioImportFailure("failed", "提示", "failed", clear_runtime=False)
            ))
            if boundary == "warning"
            else lambda *_args: _stage()
        )
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=bus,
        runtime=runtime,
        import_service=service,
        logger=logger,
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert bus.pending_import_terminal_delivery_count == 0
    assert model.active_import_id is None
    assert runtime.recorded_path == "old.wav"
    assert runtime.data_struct.store_wave_data_multi is old_multi


@pytest.mark.parametrize("boundary", ["metadata", "reference"])
@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("staging callback failed"),
        lambda: KeyboardInterrupt("staging callback interrupted"),
        lambda: SystemExit("staging callback exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_formal_staging_rechecks_owner_after_metadata_and_reference_callbacks(
    boundary, escape_factory, native_delete
):
    bus = SequenceEventBus()
    runtime = _runtime()
    old_multi = runtime.data_struct.store_wave_data_multi
    holder = {}

    def close_then_escape(*_args, **_kwargs):
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped

    def metadata(*_args, **_kwargs):
        if boundary == "metadata":
            close_then_escape()
        return {"recorded_channels": ()}

    def reference(owner, *_args, **_kwargs):
        close_then_escape()
        owner.stimulus_data = np.ones(3, dtype=np.float32)
        owner.stimulus_info = {"sample_rate": 32_000, "total_time": 3 / 32_000}
        return True

    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (
            np.zeros((3, 1), dtype=np.float32),
            32_000,
        ),
        metadata_reader=metadata,
        reference_builder=reference,
    )
    command = _command(
        mode="IMPORT_STIMULUS_AUDIO" if boundary == "reference" else "IMPORT_AUDIO"
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            clear_import_projection=lambda: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _enabled: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=service,
    )
    holder["controller"] = controller

    assert controller.handle_load_imported_audio_requested(command) is False
    assert bus.pending_import_terminal_delivery_count == 0
    assert model.active_import_id is None
    assert runtime.recorded_path == "old.wav"
    assert runtime.data_struct.store_wave_data_multi is old_multi


def test_native_deleted_import_controller_is_collectable_after_reentrant_abort():
    bus = SequenceEventBus()
    holder = {}

    def choose(selected):
        sip.delete(holder["controller"])
        return selected

    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(choose_import_audio_path=choose),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    holder["controller"] = controller
    reference = weakref.ref(controller)

    assert controller.handle_load_imported_audio_requested(_command()) is False
    del holder["controller"]
    del controller
    gc.collect()

    assert reference() is None
    assert bus.pending_import_terminal_delivery_count == 0


def test_off_thread_import_is_rejected_before_concurrent_disconnect():
    bus = SequenceEventBus()
    runtime = _runtime()
    old_multi = runtime.data_struct.store_wave_data_multi
    entered = threading.Event()
    results = []
    errors = []

    def load(*_args):
        entered.set()
        return _stage()

    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(choose_import_audio_path=lambda selected: selected),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=load),
    )

    def run_first():
        try:
            results.append(
                controller.handle_load_imported_audio_requested(_command())
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run_first)
    worker.start()
    worker.join(5)
    assert worker.is_alive() is False
    assert entered.is_set() is False
    assert controller.disconnect() is True

    assert errors == []
    assert results == [False]
    assert bus.pending_import_terminal_delivery_count == 0
    assert runtime.recorded_path == "old.wav"
    assert runtime.data_struct.store_wave_data_multi is old_multi


def test_checkpoint_capture_failure_before_mutation_preserves_old_runtime_and_projection():
    runtime = _runtime()
    old_multi = runtime.data_struct.store_wave_data_multi
    projection_calls = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=lambda: (_ for _ in ()).throw(
            KeyboardInterrupt("capture interrupted")
        ),
        present_import_warning=lambda *_args: None,
        clear_import_projection=lambda: projection_calls.append("clear"),
        set_import_data_enabled=lambda value: projection_calls.append(
            ("enabled", value)
        ),
    )
    bus = _bus()
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
        workflow_identity_provider=lambda: {
            "import_id": "import-1",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert runtime.recorded_path == "old.wav"
    assert runtime.recorded_signal_info == {"file_path": "old.wav"}
    assert runtime.data_struct.store_wave_data_multi is old_multi
    assert projection_calls == []
    assert len(bus.events.imported_audio_failed.values) == 1


def test_partial_restore_converges_to_complete_empty_state_and_attempts_every_surface():
    class FailOldMultiOnce(SimpleNamespace):
        armed = False
        failed = False

        def __setattr__(self, name, value):
            if name == "store_wave_data_multi" and self.armed and not self.failed:
                self.failed = True
                raise RuntimeError("old multi restore failed")
            super().__setattr__(name, value)

    runtime = _runtime()
    data = FailOldMultiOnce(**vars(runtime.data_struct))
    data.armed = True
    runtime.data_struct = data
    calls = []

    def fail_projection(*_args):
        raise RuntimeError("new projection failed")

    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=lambda: ("old-plot", True),
        restore_import_projection=lambda _checkpoint: (_ for _ in ()).throw(
            RuntimeError("plot restore failed")
        ),
        present_import_warning=lambda *_args: None,
        show_imported_audio=fail_projection,
        set_import_data_enabled=lambda enabled: calls.append(("enabled", enabled)),
        clear_import_projection=lambda: calls.append("plot-cleared"),
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=_bus(),
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
        workflow_identity_provider=lambda: {
            "import_id": "import-1",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert runtime.recorded_path is None
    assert runtime.recorded_signal_info is None
    for field in (
        "store_wave_data_multi",
        "store_wave_data",
        "sample_rate",
        "audio_lenth",
        "stimulus_data",
        "stimulus_info",
        "wav_calibration_metadata",
    ):
        assert getattr(data, field) is None
    assert data.wav_calibration_metadata_authoritative is False
    assert data.wav_calibration_warning_shown is False
    assert "plot-cleared" in calls
    assert ("enabled", False) in calls
    assert controller.model.import_runtime_consistent is True


def test_unrecoverable_clear_marks_inconsistent_and_blocks_next_import():
    class RefuseEmptyPath:
        def __init__(self):
            self.data_struct = _runtime().data_struct
            self._recorded_path = "old.wav"
            self.recorded_signal_info = {"file_path": "old.wav"}

        @property
        def recorded_path(self):
            return self._recorded_path

        @recorded_path.setter
        def recorded_path(self, value):
            if value is None:
                raise RuntimeError("cannot clear identity")
            self._recorded_path = value

    runtime = RefuseEmptyPath()
    loads = []
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        capture_import_projection=lambda: ("old", True),
        restore_import_projection=lambda _checkpoint: (_ for _ in ()).throw(
            RuntimeError("cannot restore projection")
        ),
        present_import_warning=lambda *_args: None,
        show_imported_audio=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("projection failed")
        ),
        clear_import_projection=lambda: None,
        set_import_data_enabled=lambda _value: None,
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=_bus(),
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda *_args: loads.append(True) or _stage()
        ),
        workflow_identity_provider=lambda: {
            "import_id": "import-1",
            "phase": "IMPORTING",
        },
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert model.import_runtime_consistent is False
    assert "recorded_path" in model.import_consistency_failure
    assert controller.handle_load_imported_audio_requested(
        _command("import-2")
    ) is False
    assert loads == [True]


def test_silently_ignored_clear_is_diagnosed_as_inconsistent():
    class IgnoreEmptyPath:
        def __init__(self):
            self.data_struct = _runtime().data_struct
            self._recorded_path = "old.wav"
            self.recorded_signal_info = {"file_path": "old.wav"}

        @property
        def recorded_path(self):
            return self._recorded_path

        @recorded_path.setter
        def recorded_path(self, value):
            if value is not None:
                self._recorded_path = value

    runtime = IgnoreEmptyPath()
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: "old",
            restore_import_projection=lambda _checkpoint: False,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("projection failed")
            ),
            clear_import_projection=lambda: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=_bus(),
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert runtime.recorded_path == "old.wav"
    assert controller.model.import_runtime_consistent is False
    assert "runtime.recorded_path" in controller.model.import_consistency_failure


@pytest.mark.parametrize(
    ("decoded", "sample_rate"),
    [
        (np.empty(0, dtype=np.float32), 48_000),
        (np.empty((1, 0), dtype=np.float32), 48_000),
        (np.zeros((1, 1, 1), dtype=np.float32), 48_000),
        (np.array([0.0, np.nan], dtype=np.float32), 48_000),
        (np.array([0.0, np.inf], dtype=np.float32), 48_000),
        (np.zeros(2, dtype=np.float32), 0),
        (np.zeros(2, dtype=np.float32), float("nan")),
    ],
    ids=["empty-1d", "empty-2d", "three-dimensional", "nan", "inf", "zero-rate", "nan-rate"],
)
def test_import_service_rejects_invalid_decoded_audio(decoded, sample_rate):
    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (decoded, sample_rate),
        metadata_reader=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(AudioImportFailure):
        service.load(_command(), "new.wav")


def test_import_stage_owns_contiguous_readonly_audio_and_detached_metadata():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    metadata = {"nested": [{"gain": 2.0}]}
    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (source, 48_000),
        metadata_reader=lambda *_args, **_kwargs: metadata,
    )

    stage = service.load(_command(), "new.wav")
    expected_multi = stage.audio_multi.copy()
    expected_mono = stage.audio_mono.copy()
    source[:] = -1
    metadata["nested"][0]["gain"] = 99

    assert np.array_equal(stage.audio_multi, expected_multi)
    assert np.array_equal(stage.audio_mono, expected_mono)
    assert stage.audio_multi.flags.owndata and stage.audio_multi.flags.c_contiguous
    assert stage.audio_mono.flags.owndata and stage.audio_mono.flags.c_contiguous
    assert stage.audio_multi.flags.writeable is False
    assert stage.audio_mono.flags.writeable is False
    assert stage.calibration_metadata["nested"][0]["gain"] == 2.0
    with pytest.raises(TypeError):
        stage.calibration_metadata["new"] = True


@pytest.mark.parametrize(
    "metadata",
    [type("HostileDict", (dict,), {})(ok=True), {"values": list(range(5000))}],
    ids=["hostile-mapping", "oversized"],
)
def test_import_service_rejects_hostile_or_oversized_metadata(metadata):
    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (np.zeros(4), 48_000),
        metadata_reader=lambda *_args, **_kwargs: metadata,
    )

    with pytest.raises(AudioImportFailure):
        service.load(_command(), "new.wav")


def test_import_reference_is_whitelisted_detached_and_readonly():
    stimulus = np.arange(4, dtype=np.float32)
    info = {"nested": ["stable"]}

    def build(reference, *_args, **_kwargs):
        reference.stimulus_data = stimulus
        reference.stimulus_info = info
        reference.alignment_sample_count = 4
        reference.total_time = 0.1
        reference.hostile_extra = object()
        return True

    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (np.zeros(4), 48_000),
        metadata_reader=lambda *_args, **_kwargs: None,
        reference_builder=build,
    )
    stage = service.load(
        _command(mode="IMPORT_STIMULUS_AUDIO"), "new.wav"
    )
    stimulus[:] = -1
    info["nested"].append("changed")

    assert np.array_equal(stage.reference.stimulus_data, np.arange(4))
    assert stage.reference.stimulus_data.flags.writeable is False
    assert stage.reference.stimulus_info["nested"] == ("stable",)
    assert not hasattr(stage.reference, "hostile_extra")


@pytest.mark.parametrize("cancel_point", ["picker", "loader", "projection"])
def test_canonical_import_cancel_first_wins_at_each_synchronous_checkpoint(cancel_point):
    runtime = _runtime()
    old_path = runtime.recorded_path
    bus = _bus()
    workflow = {"import_id": "import-1", "phase": "IMPORTING", "workflow_generation": 7}
    calls = []
    controller = None

    def cancel():
        workflow["phase"] = "CANCELLING"
        command = messages.CancelImportedAudioRequested(
            "import-1", 7, "operator cancelled"
        )
        calls.append(("cancel", controller.handle_cancel_imported_audio_requested(command)))
        calls.append(("duplicate", controller.handle_cancel_imported_audio_requested(command)))

    def choose(selected):
        if cancel_point == "picker":
            cancel()
        return selected

    def load(*_args):
        if cancel_point == "loader":
            cancel()
        return _stage()

    def project(*_args):
        if cancel_point == "projection":
            cancel()

    view = SimpleNamespace(
        choose_import_audio_path=choose,
        capture_import_projection=lambda: ("old-plot", True),
        restore_import_projection=lambda _checkpoint: calls.append("restore"),
        present_import_warning=lambda *_args: None,
        show_imported_audio=project,
        set_import_data_enabled=lambda _value: None,
        clear_import_projection=lambda: calls.append("clear"),
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        view,
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(load=load),
        workflow_identity_provider=lambda: workflow,
    )

    command = LoadImportedAudioRequested(
        "command-1", "import-1", "IMPORT_AUDIO", "new.wav", _configuration(),
        workflow_generation=7,
    )
    assert controller.handle_load_imported_audio_requested(command) is False
    assert calls[0:2] == [("cancel", True), ("duplicate", False)]
    assert runtime.recorded_path == old_path
    assert bus.events.imported_audio_ready.values == []
    assert len(bus.events.imported_audio_failed.values) == 1
    assert "cancel" in bus.events.imported_audio_failed.values[0].reason


def test_import_cancel_rejects_wrong_identity_and_generation():
    model = RecordingModel()
    model.begin_import("import-1", workflow_generation=7)
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(),
        bus=_bus(),
        runtime=_runtime(),
    )

    assert controller.handle_cancel_imported_audio_requested(
        messages.CancelImportedAudioRequested("wrong", 7, "cancel")
    ) is False
    assert controller.handle_cancel_imported_audio_requested(
        messages.CancelImportedAudioRequested("import-1", 8, "cancel")
    ) is False
    assert model.import_cancel_pending is False


@pytest.mark.parametrize(
    "escape_factory",
    [
        lambda: None,
        lambda: RuntimeError("cancel identity failed"),
        lambda: KeyboardInterrupt("cancel identity interrupted"),
        lambda: SystemExit("cancel identity exited"),
    ],
    ids=["success", "ordinary", "keyboard-interrupt", "system-exit"],
)
@pytest.mark.parametrize("native_delete", [False, True], ids=["disconnect", "sip-delete"])
def test_cancel_identity_callback_close_never_marks_cancel_pending(
    escape_factory, native_delete
):
    model = RecordingModel()
    model.begin_import("import-1", workflow_generation=7)
    holder = {}

    def identity():
        controller = holder["controller"]
        if native_delete:
            sip.delete(controller)
        else:
            assert controller.disconnect() is True
        escaped = escape_factory()
        if escaped is not None:
            raise escaped
        return {
            "import_id": "import-1",
            "phase": "CANCELLING",
            "workflow_generation": 7,
        }

    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(),
        bus=_bus(),
        runtime=_runtime(),
        workflow_identity_provider=identity,
    )
    holder["controller"] = controller

    assert controller.handle_cancel_imported_audio_requested(
        messages.CancelImportedAudioRequested("import-1", 7, "cancel")
    ) is False
    assert model.import_cancel_pending is False


@pytest.mark.parametrize(
    "delivery_failure", [RuntimeError("ordinary"), KeyboardInterrupt("interrupt"), SystemExit("exit")],
    ids=["ordinary", "keyboard-interrupt", "system-exit"],
)
def test_import_terminal_requires_critical_ack_and_retries_same_event(delivery_failure):
    runtime = _runtime()
    raw = _Signal()
    deliveries = []
    accept = {"value": False}

    class ReliableBus:
        def __init__(self):
            self.events = SimpleNamespace(
                imported_audio_ready=raw, imported_audio_failed=_Signal()
            )

        def has_import_terminal_recipients(self):
            return True

        def deliver_import_terminal(self, delivery_id, event):
            deliveries.append((delivery_id, event))
            if not accept["value"]:
                raise delivery_failure
            return True

    bus = ReliableBus()
    loads = []
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=runtime,
        import_service=SimpleNamespace(
            load=lambda *_args: loads.append(True) or _stage()
        ),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert model.active_import_id == "import-1"
    assert controller.pending_import_terminal_identity == "import-1"
    assert raw.values == []
    model.set_import_consistency(False, "previous rollback incomplete")
    assert controller.handle_load_imported_audio_requested(
        _command("import-2")
    ) is False
    assert loads == [True]
    accept["value"] = True
    assert controller.retry_pending_import_terminal("import-1") is True
    assert model.active_import_id is None
    assert controller.pending_import_terminal_identity is None
    assert raw.values and len(raw.values) == 1
    assert deliveries[0][1] is deliveries[1][1]
    assert controller.retry_pending_import_terminal("import-1") is False


def test_formal_bus_without_workflow_recipient_fails_closed_and_keeps_terminal_pending():
    bus = SequenceEventBus()
    raw = []
    bus.events.imported_audio_ready.connect(raw.append)
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert controller.pending_import_terminal_identity == "import-1"
    assert controller.model.active_import_id == "import-1"
    assert raw == []


def test_pending_terminal_uses_one_bounded_backoff_timer_and_exact_event():
    attempts = []
    outcomes = [
        RuntimeError("workflow unavailable"),
        KeyboardInterrupt("workflow interrupted"),
        SystemExit("workflow exited"),
        *([False] * 8),
        True,
    ]

    class RetryBus:
        def __init__(self):
            self.events = SimpleNamespace(
                imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
            )

        def has_import_terminal_recipients(self):
            return True

        def deliver_import_terminal(self, delivery_id, event):
            attempts.append((delivery_id, event))
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=RetryBus(),
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    timer = controller.import_terminal_retry_timer
    first_delay = controller.import_terminal_retry_delay_ms
    assert timer.isSingleShot() and timer.isActive()
    assert 0 < first_delay <= controller.import_terminal_retry_max_delay_ms
    controller._schedule_import_terminal_retry()
    assert controller.import_terminal_retry_timer is timer
    assert controller.import_terminal_retry_delay_ms == first_delay

    delays = [first_delay]
    while len(outcomes) > 1:
        controller.handle_import_terminal_retry_timeout()
        delays.append(controller.import_terminal_retry_delay_ms)
    assert delays == sorted(delays)
    assert delays[-1] == controller.import_terminal_retry_max_delay_ms
    controller.handle_import_terminal_retry_timeout()

    assert controller.pending_import_terminal_identity is None
    assert model.active_import_id is None
    assert timer.isActive() is False
    assert len({id(event) for _delivery_id, event in attempts}) == 1
    attempt_count = len(attempts)
    controller.handle_import_terminal_retry_timeout()
    assert len(attempts) == attempt_count


def test_permanently_rejected_import_terminal_does_not_schedule_retry_timer():
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow",
        lambda _event: ImportTerminalRecipientResult.PERMANENT_REJECT,
        critical=True,
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert controller.pending_import_terminal_identity is None
    assert controller.import_terminal_retry_timer.isActive() is False
    assert model.active_import_id is None
    assert bus.import_terminal_abandonment_reason(
        ("ImportedAudioReady", "import-1")
    ) == "recipient-permanent-reject"


def test_recording_disconnect_abandons_pending_terminal_without_retiring_identity():
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    assert controller.handle_load_imported_audio_requested(_command()) is False
    delivery_id = ("ImportedAudioReady", "import-1")

    assert controller.disconnect() is True
    assert controller.import_terminal_retry_timer.isActive() is False
    assert controller.model.active_import_id == "import-1"
    assert bus.import_terminal_abandonment_reason(delivery_id) == "recording-disconnect"
    assert controller.disconnect() is False


def test_recording_destroy_abandons_pending_terminal_without_false_ack():
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    model = RecordingModel()
    controller = SequenceRecordingImportController(
        model,
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )
    assert controller.handle_load_imported_audio_requested(_command()) is False

    controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    _QAPP.processEvents()

    assert model.active_import_id == "import-1"
    assert bus.import_terminal_abandonment_reason(
        ("ImportedAudioReady", "import-1")
    ) == "recording-disconnect"


@pytest.mark.parametrize(
    ("alignment", "expected_present", "expected_value"),
    [("absent", False, None), (None, True, None), (4, True, 4)],
    ids=["absent", "present-none", "present-value"],
)
def test_import_reference_preserves_alignment_presence(
    alignment, expected_present, expected_value
):
    def build(reference, *_args, **_kwargs):
        reference.stimulus_data = np.zeros(4, dtype=np.float32)
        reference.stimulus_info = {"sample_rate": 48_000}
        if alignment != "absent":
            reference.alignment_sample_count = alignment
        return True

    service = SequenceImportedAudioService(
        audio_loader=lambda *_args, **_kwargs: (np.zeros(4), 48_000),
        metadata_reader=lambda *_args, **_kwargs: None,
        reference_builder=build,
    )

    stage = service.load(
        _command(mode="IMPORT_STIMULUS_AUDIO"), "new.wav"
    )

    assert stage.reference.alignment_sample_count_present is expected_present
    assert stage.reference.alignment_sample_count == expected_value


@pytest.mark.parametrize(
    ("alignment", "expected_present", "expected_value"),
    [(None, False, None), (4, True, 4)],
    ids=["none-removes", "value-sets"],
)
def test_import_commit_preserves_legacy_alignment_hasattr_semantics(
    alignment, expected_present, expected_value
):
    runtime = _runtime()
    runtime.data_struct.alignment_sample_count = 99
    reference = SimpleNamespace(
        stimulus_data=np.zeros(3, dtype=np.float32),
        stimulus_info={},
        alignment_sample_count=alignment,
        alignment_sample_count_present=True,
    )
    stage = _stage()
    stage = ImportedAudioStage(
        stage.file_path,
        "IMPORT_STIMULUS_AUDIO",
        stage.sample_rate,
        stage.audio_multi,
        stage.audio_mono,
        stage.sample_count,
        stage.calibration_metadata,
        reference,
    )
    controller = SequenceRecordingImportController(
        RecordingModel(),
        SimpleNamespace(
            choose_import_audio_path=lambda selected: selected,
            capture_import_projection=lambda: None,
            present_import_warning=lambda *_args: None,
            show_imported_audio=lambda *_args: None,
            set_import_data_enabled=lambda _value: None,
        ),
        bus=_bus(),
        runtime=runtime,
        import_service=SimpleNamespace(load=lambda *_args: stage),
    )

    assert controller.handle_load_imported_audio_requested(
        _command(mode="IMPORT_STIMULUS_AUDIO")
    ) is True
    assert hasattr(runtime.data_struct, "alignment_sample_count") is expected_present
    if expected_present:
        assert runtime.data_struct.alignment_sample_count == expected_value


def test_recording_view_import_recovery_ports_propagate_callback_acknowledgement():
    view = SequenceRecordingImportView(
        import_projection_restorer=lambda _state: False,
        import_plot_projection_restorer=lambda _state: "plot-ack",
        clear_import_plot=lambda: "clear-ack",
        import_data_enabled_setter=lambda _enabled: False,
    )

    assert view.restore_import_projection("state") is False
    assert view.restore_import_plot("plot") == "plot-ack"
    assert view.clear_import_projection() == "clear-ack"
    assert view.set_import_data_enabled(True) is False


def test_initial_import_enabled_negative_ack_enters_transaction_recovery():
    model = RecordingModel()
    enabled_calls = []
    view = SequenceRecordingImportView(
        import_projection_capturer=lambda: ("old-plot", False),
        import_plot_projection_restorer=lambda _state: None,
        clear_import_plot=lambda: None,
        plot_imported_audio=lambda *_args: None,
        import_data_enabled_setter=lambda enabled: enabled_calls.append(enabled) or False,
        warning_presenter=lambda *_args: None,
    )
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=_bus(),
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert enabled_calls[0] is True
    assert model.import_runtime_consistent is False
    assert model.active_import_id is None


@pytest.mark.parametrize(
    "outcome_factory",
    [
        lambda: False,
        lambda: RuntimeError("view failed"),
        lambda: KeyboardInterrupt("view interrupted"),
        lambda: SystemExit("view exited"),
    ],
    ids=["false", "ordinary", "keyboard-interrupt", "system-exit"],
)
def test_view_recovery_negative_ack_fails_closed_without_terminal_retirement(
    outcome_factory,
):
    calls = []

    def callback(name):
        def invoke(*_args):
            calls.append(name)
            outcome = outcome_factory()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return invoke

    model = RecordingModel()
    view = SequenceRecordingImportView(
        import_projection_capturer=lambda: ("old-plot", True),
        import_plot_projection_restorer=callback("restore-plot"),
        clear_import_plot=callback("clear-plot"),
        plot_imported_audio=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("plot failed")
        ),
        import_data_enabled_setter=callback("enabled"),
        warning_presenter=lambda *_args: None,
    )
    bus = SequenceEventBus()
    bus.register_import_terminal_recipient(
        "workflow", lambda _event: False, critical=True
    )
    controller = SequenceRecordingImportController(
        model,
        view,
        bus=bus,
        runtime=_runtime(),
        import_service=SimpleNamespace(load=lambda *_args: _stage()),
    )

    assert controller.handle_load_imported_audio_requested(_command()) is False
    assert model.import_runtime_consistent is False
    assert model.active_import_id == "import-1"
    assert controller.pending_import_terminal_identity == "import-1"
    assert "restore-plot" in calls
    assert "clear-plot" in calls
    assert "enabled" in calls
    controller.disconnect()
