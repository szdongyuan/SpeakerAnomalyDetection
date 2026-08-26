from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

from base.load_config import LoadUiConfig
from base.shortcut_trigger_manager import ShortcutTriggerManager
from base.unified_hid_device_manager import UnifiedHardwareManager
from ui.sequence.barcode_router import BarcodeRouter
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_legacy_recording_bridge import (
    LegacyRecordingAdmissionBridge,
    legacy_recording_session_snapshot,
)
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    ReplayRequested,
    StartTestRequested,
)
from ui.sequence.sequence_resource_lifecycle_controller import (
    SequenceResourceLifecycleController,
    SequenceResourceLifecycleView,
    _CANONICAL_TCP_MIRROR_STATE,
)
from ui.sequence.sequence_trigger_controller import SequenceTriggerController
from ui.sequence.sequence_trigger_model import (
    SequenceTriggerModel,
    TcpIdentityOutboxCapacityError,
)
from ui.sequence.sequence_trigger_view import SequenceTriggerView
from ui.sequence.sequence_trigger_resource_lifecycle_port import (
    SequenceTriggerResourceLifecyclePort,
)
from ui.sequence.sn_regex_manage_dialog import SnRegexManageDialog
from ui.sequence.sequence_workflow_controller import SequenceWorkflowController
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel, WorkflowPhase
from ui.tcp_config_dialog import TcpConfigDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Logger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(("debug", message))

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _View:
    def __init__(self, *, serial="SN-OLD", scanner=True):
        self.serial = serial
        self.scanner = scanner
        self.serial_enabled = True
        self.serial_read_only = False
        self.focus = "serial"
        self.product_input = object()
        self.count_input = object()
        self.serial_input = object()
        self.invalid = []
        self.regex_rejections = []
        self.mode_rejections = []
        self.busy_rejections = []
        self.workflow_rejections = []
        self.focus_calls = 0
        self.close_analysis_calls = 0
        self.tcp_states = []
        self.scanner_states = []
        self.tcp_dialog_result = None
        self.close_dialog_calls = 0

    def is_scanner_checked(self):
        return self.scanner

    def is_serial_enabled(self):
        return self.serial_enabled

    def serial_text(self):
        return self.serial

    def set_serial_text(self, text):
        self.serial = text

    def clear_serial_text(self):
        self.serial = ""

    def focus_widget(self):
        return self.focus

    def focus_serial_input(self, *, select_all=False):
        self.focus_calls += 1
        self.focus = "serial"

    def prepare_for_continuous_scan(self):
        self.close_analysis_calls += 1

    def show_invalid_barcode(self, barcode, invalid_chars):
        self.invalid.append((barcode, tuple(invalid_chars)))

    def show_regex_rejection(self, rule, sn_text, value_label, retry_hint):
        self.regex_rejections.append(
            (dict(rule), sn_text, value_label, retry_hint)
        )

    def show_mode_rejection(self, trigger_source, mode):
        self.mode_rejections.append((trigger_source, mode))

    def show_busy_rejection(self, trigger_source):
        self.busy_rejections.append(trigger_source)

    def show_workflow_rejection(self, reason):
        self.workflow_rejections.append(reason)

    def set_scanner_enabled(self, enabled):
        self.scanner_states.append(bool(enabled))

    def set_serial_read_only(self, read_only):
        self.serial_read_only = bool(read_only)

    def present_tcp_state(self, enabled):
        self.tcp_states.append(bool(enabled))

    def open_tcp_dialog(
        self, enabled, host, port, on_accepted=None, on_rejected=None
    ):
        self.tcp_dialog_callbacks = (on_accepted, on_rejected)
        return True

    def close_dialogs(self):
        self.close_dialog_calls += 1


class _Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


class _Timer:
    def __init__(self):
        self.starts = []
        self.stops = 0
        self.active = False

    def start(self, interval=None):
        self.starts.append(interval)
        self.active = True

    def stop(self):
        self.stops += 1
        self.active = False

    def isActive(self):
        return self.active


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)

    def emit(self, *args):
        for slot in tuple(self.slots):
            slot(*args)


class _KeyEvent:
    def __init__(self, key, text="", modifiers=Qt.NoModifier):
        self._key = key
        self._text = text
        self._modifiers = modifiers

    def type(self):
        return QEvent.KeyPress

    def key(self):
        return self._key

    def text(self):
        return self._text

    def modifiers(self):
        return self._modifiers


def _controller(
    *,
    model=None,
    view=None,
    clock=None,
    emitted=None,
    generation=lambda: 7,
    workflow_active=lambda: False,
    mode_available=lambda: True,
    mode=lambda: "RECORD_ONLY",
    regex_rule=lambda: {"name": "default", "pattern": r".*"},
    timer=None,
    **kwargs,
):
    model = model or SequenceTriggerModel()
    view = view or _View()
    clock = clock or _Clock()
    emitted = emitted if emitted is not None else []
    command_id_factory = kwargs.pop(
        "command_id_factory",
        lambda counter=iter(range(1000)): f"trigger-{next(counter)}",
    )
    logger = kwargs.pop("logger", _Logger())
    controller = SequenceTriggerController(
        model,
        view,
        start_publisher=emitted.append,
        configuration_generation_provider=generation,
        workflow_active_provider=workflow_active,
        external_mode_available_provider=mode_available,
        acquisition_mode_provider=mode,
        regex_rule_loader=regex_rule,
        monotonic=clock,
        command_id_factory=command_id_factory,
        debounce_timer=timer,
        logger=logger,
        **kwargs,
    )
    return controller, model, view, clock, emitted


def _composed_tcp_controller(qapp, **kwargs):
    assert _CANONICAL_TCP_MIRROR_STATE.write(None) is True
    owner = QWidget()
    lifecycle = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(owner), parent=owner
    )
    kwargs.setdefault("tcp_server_factory", _TcpServer)
    controller, model, view, _clock, _emitted = _controller(
        tcp_mirror_getter=lifecycle.read_tcp_mirror_identity,
        tcp_mirror_setter=lifecycle.write_tcp_mirror_identity,
        **kwargs,
    )
    lifecycle.tcp_resource_port = SequenceTriggerResourceLifecyclePort(
        controller
    )
    return owner, lifecycle, controller, model, view


def _retire_composed_tcp_owner(lifecycle):
    token = lifecycle._tcp_mirror_owner_token
    assert _CANONICAL_TCP_MIRROR_STATE.unregister(lifecycle, token) is True
    assert _CANONICAL_TCP_MIRROR_STATE.write(None) is True


def _drain_events(qapp, rounds=8):
    for _ in range(rounds):
        qapp.processEvents()


def _flush_deferred_deletes(qapp):
    for _ in range(3):
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
    gc.collect()


class _AsyncMessageBox(QDialog):
    """Real Qt dialog with the QMessageBox API needed by the trigger view."""

    Warning = QMessageBox.Warning
    Ok = QMessageBox.Ok

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""

    def setIcon(self, _icon):
        return None

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setStandardButtons(self, _buttons):
        return None

    def open(self):
        self.show()


def test_model_owns_trigger_capture_dedup_shortcut_and_tcp_state():
    model = SequenceTriggerModel()

    model.barcode_capture_buffer = "SN-1234567"
    model.barcode_first_char_ts = 1.0
    model.last_committed_barcode = "SN-1234567"
    model.shortcut_processing = True
    model.external_trigger_available = False
    model.tcp_enabled = True
    model.tcp_connected = True

    model.reset_capture(clear_dedup=False)

    assert model.barcode_capture_buffer == ""
    assert model.barcode_first_char_ts is None
    assert model.last_committed_barcode == "SN-1234567"
    assert model.shortcut_processing is True
    assert model.external_trigger_available is False
    assert model.tcp_enabled is True
    assert model.tcp_connected is True


def test_model_constructor_preserves_controlled_tcp_server_compatibility():
    server = object()

    model = SequenceTriggerModel(tcp_server=server)

    assert model.tcp_server is server


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, ""), ("  SN-123  ", "SN-123"), (123, "123")],
)
def test_normalization_matches_existing_behavior(raw, expected):
    controller, *_ = _controller()
    assert controller.normalize_barcode(raw) == expected


def test_invalid_filename_characters_are_reported_in_input_order():
    controller, *_ = _controller()
    assert controller.barcode_invalid_characters(r"SN:/?A:") == (":", "/", "?", ":")


def test_valid_barcode_publishes_exact_immutable_start_payload():
    controller, model, view, _clock, emitted = _controller(
        regex_rule=lambda: {"name": "sn", "pattern": r"SN-\d{3}"}
    )

    assert controller.commit_barcode("  SN-123  ", source="hid") is True

    assert emitted == [
        StartTestRequested(
            command_id="trigger-0",
            source="hid",
            label="not_labeled",
            skip_sn_regex_validation=False,
            configuration_generation=7,
        )
    ]
    assert view.serial == "SN-123"
    assert view.close_analysis_calls == 1
    assert model.last_committed_barcode == "SN-123"
    assert model.last_committed_barcode_time == 100.0


def test_invalid_barcode_clears_input_and_dedup_so_immediate_retry_is_allowed():
    controller, model, view, _clock, emitted = _controller()

    assert controller.commit_barcode("BAD/SN", source="hid") is False

    assert emitted == []
    assert view.invalid == [("BAD/SN", ("/",))]
    assert view.serial == ""
    assert model.last_committed_barcode is None
    assert model.barcode_capture_buffer == ""


def test_regex_rejection_uses_selected_rule_and_allows_immediate_retry():
    rule = {"name": "line-a", "pattern": r"SN-\d{3}"}
    controller, model, view, _clock, emitted = _controller(regex_rule=lambda: rule)

    assert controller.commit_barcode("BAD-SN", source="wedge_enter") is False

    assert emitted == []
    assert view.regex_rejections == [
        (
            rule,
            "BAD-SN",
            "实际扫码内容",
            "请检查扫码内容或切换正确规则后重新扫码。",
        )
    ]
    assert model.last_committed_barcode is None


def test_busy_rejection_happens_before_serial_or_dedup_mutation():
    controller, model, view, _clock, emitted = _controller(
        workflow_active=lambda: True
    )

    assert controller.commit_barcode("SN-NEW", source="hid") is False

    assert emitted == []
    assert view.serial == "SN-OLD"
    assert view.busy_rejections == ["扫码枪"]
    assert model.last_committed_barcode is None


def test_unsupported_mode_rejects_external_trigger_and_clears_scanner_sn():
    controller, model, view, _clock, emitted = _controller(
        mode_available=lambda: False,
        mode=lambda: "IMPORT_AUDIO",
    )

    assert controller.commit_barcode("SN-123", source="hid") is False

    assert emitted == []
    assert view.serial == ""
    assert view.mode_rejections == [("扫码枪", "IMPORT_AUDIO")]
    assert model.last_committed_barcode is None


def test_hid_and_wedge_paths_are_deduplicated_without_losing_first_start():
    controller, model, _view, clock, emitted = _controller()

    assert controller.handle_hid_barcode("SN-1234567") is True
    clock.value += 0.1
    assert controller.commit_barcode("SN-1234567", source="wedge_enter") is False

    assert len(emitted) == 1
    assert emitted[0].source == "hid"
    assert model.hid_mode_active_until == pytest.approx(101.0)


def test_line_edit_debounce_commits_fast_scanner_input():
    timer = _Timer()
    view = _View(serial="SN-1234567")
    controller, model, _view, clock, emitted = _controller(
        view=view, timer=timer
    )

    controller.handle_barcode_text_changed("S")
    clock.value += 0.2
    controller.handle_barcode_text_changed("SN-1234567")
    controller.handle_barcode_debounce_timeout()

    assert timer.starts == [model.debounce_interval_ms, model.debounce_interval_ms]
    assert len(emitted) == 1
    assert emitted[0].source == "wedge_debounce"


def test_slow_or_short_debounce_input_does_not_start():
    view = _View(serial="ABC")
    controller, _model, _view, clock, emitted = _controller(view=view)

    controller.handle_barcode_text_changed("A")
    clock.value += 1.0
    controller.handle_barcode_text_changed("ABC")
    assert controller.handle_barcode_debounce_timeout() is False

    assert emitted == []


def test_continuous_scan_is_allowed_after_workflow_returns_idle_and_dedup_expires():
    active = {"value": False}
    controller, _model, _view, clock, emitted = _controller(
        workflow_active=lambda: active["value"]
    )

    assert controller.commit_barcode("SN-1234567", source="hid") is True
    active["value"] = True
    clock.value += 1.0
    assert controller.commit_barcode("SN-7654321", source="hid") is False
    active["value"] = False
    assert controller.commit_barcode("SN-7654321", source="hid") is True

    assert [command.label for command in emitted] == ["not_labeled", "not_labeled"]


def test_shortcut_reentry_is_rejected_while_first_publication_is_in_progress():
    model = SequenceTriggerModel()
    view = _View(scanner=False)
    emitted = []
    holder = {}

    def publish(command):
        emitted.append(command)
        assert holder["controller"].handle_shortcut_trigger() is False

    controller = SequenceTriggerController(
        model,
        view,
        start_publisher=publish,
        configuration_generation_provider=lambda: 4,
        workflow_active_provider=lambda: False,
        external_mode_available_provider=lambda: True,
        acquisition_mode_provider=lambda: "RECORD_ONLY",
        command_id_factory=lambda: "shortcut-command",
        logger=_Logger(),
    )
    holder["controller"] = controller

    assert controller.handle_shortcut_trigger() is True

    assert len(emitted) == 1
    assert emitted[0].source == "shortcut"
    assert model.shortcut_processing is False


def test_optical_trigger_publishes_without_touching_serial():
    controller, _model, view, _clock, emitted = _controller()

    assert controller.handle_optical_trigger() is True

    assert view.serial == "SN-OLD"
    assert emitted[0].source == "optical"
    assert emitted[0].label == "not_labeled"


def test_manual_request_honors_regex_while_tcp_explicit_skip_bypasses_it():
    controller, _model, view, _clock, emitted = _controller(
        regex_rule=lambda: {"name": "sn", "pattern": r"SN-\d{3}"}
    )
    view.serial = "BAD"

    assert controller.request_start(label="OK", source="manual") is False
    assert controller.handle_tcp_run_test(label="NG", skip_sn_regex_validation=True) is True

    assert len(view.regex_rejections) == 1
    assert emitted == [
        StartTestRequested("trigger-1", "tcp", "NG", True, 7)
    ]


def test_start_publication_failure_does_not_commit_barcode_projection_or_dedup():
    model = SequenceTriggerModel()
    view = _View(serial="SN-OLD")
    controller = SequenceTriggerController(
        model,
        view,
        start_publisher=lambda _message: (_ for _ in ()).throw(RuntimeError("closed")),
        configuration_generation_provider=lambda: 1,
        workflow_active_provider=lambda: False,
        external_mode_available_provider=lambda: True,
        acquisition_mode_provider=lambda: "RECORD_ONLY",
        regex_rule_loader=lambda: {"name": "default", "pattern": r".*"},
        command_id_factory=lambda: "command",
        logger=_Logger(),
    )

    with pytest.raises(RuntimeError, match="closed"):
        controller.commit_barcode("SN-NEW", source="hid")

    assert view.serial == "SN-OLD"
    assert model.last_committed_barcode is None


def test_barcode_router_depends_only_on_narrow_trigger_port():
    calls = []
    port = SimpleNamespace(
        handle_barcode_return_pressed=lambda: calls.append("return") or True,
        handle_barcode_text_changed=lambda text: calls.append(("text", text)),
        handle_barcode_debounce_timeout=lambda: calls.append("timeout") or True,
        handle_keypress=lambda obj, event: calls.append((obj, event)) or True,
    )
    router = BarcodeRouter(port)
    event = object()

    assert router.on_barcode_return_pressed() is True
    router.on_barcode_text_changed("SN")
    assert router.on_barcode_debounce_timeout() is True
    assert router.handle_keypress("obj", event) is True

    assert calls == ["return", ("text", "SN"), "timeout", ("obj", event)]
    assert not hasattr(router, "ctx")


def test_trigger_domain_has_no_recording_controller_or_sequence_window_backdoor():
    root = Path(__file__).resolve().parents[2]
    controller_source = (
        root / "ui" / "sequence" / "sequence_trigger_controller.py"
    ).read_text(encoding="utf-8")
    router_source = (root / "ui" / "sequence" / "barcode_router.py").read_text(
        encoding="utf-8"
    )

    assert "start_this_play" not in controller_source
    assert "recording_controller" not in controller_source
    assert "workflow_controller" not in controller_source
    assert "from ui.sequence.sequence_widget" not in router_source
    assert "self.ctx" not in router_source


def test_global_wedge_capture_uses_model_buffer_and_enter_commits():
    controller, model, view, clock, emitted = _controller()
    view.focus = None

    for char in "SN-1234567":
        assert controller.handle_keypress(None, _KeyEvent(ord(char), char)) is True
        clock.value += 0.01
    assert controller.handle_keypress(None, _KeyEvent(Qt.Key_Return)) is True

    assert len(emitted) == 1
    assert emitted[0].source == "wedge_global_enter"
    assert model.barcode_capture_buffer == ""


def test_manual_edit_key_sets_guard_and_cancels_debounce():
    timer = _Timer()
    controller, model, view, _clock, emitted = _controller(timer=timer)
    view.focus = view.serial_input
    model.barcode_first_char_ts = 1.0
    model.barcode_last_char_ts = 1.1

    assert controller.handle_keypress(view.serial_input, _KeyEvent(Qt.Key_Backspace)) is None

    assert model.sn_textchange_manual_guard is True
    assert model.barcode_first_char_ts is None
    assert model.barcode_last_char_ts is None
    assert timer.stops == 1
    assert emitted == []


class _TcpServer:
    def __init__(self, host=None, port=None, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.start_calls = 0
        self.stop_calls = 0
        self.client_address = ("127.0.0.1", 5000)

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def test_composed_tcp_enable_accepts_own_mirror_admission_and_presents_lock(
    qapp,
):
    owner, lifecycle, controller, model, view = _composed_tcp_controller(qapp)
    try:
        assert controller.set_tcp_enabled(
            True, host="127.0.0.1", port=9001
        ) is True
        server = model.tcp_server
        assert server is not None
        assert lifecycle.read_tcp_mirror_identity() is server
        assert server.start_calls == 1
        assert view.tcp_states[-1] is True
    finally:
        controller.stop_tcp()
        _retire_composed_tcp_owner(lifecycle)
        owner.deleteLater()


def test_composed_tcp_mirror_write_correlates_own_admission(qapp):
    owner, lifecycle, controller, _model, _view = _composed_tcp_controller(
        qapp
    )
    candidate = object()
    try:
        initial_epoch = controller._resource_identity_epoch

        assert controller.tcp_mirror_setter(candidate) is True

        assert lifecycle.read_tcp_mirror_identity() is candidate
        assert controller._resource_identity_epoch == initial_epoch + 1
    finally:
        _retire_composed_tcp_owner(lifecycle)
        owner.deleteLater()


def test_tcp_mirror_overlap_rejects_second_write_without_replacing_correlation():
    mirror = {"value": None}
    first_requested = object()
    second_requested = object()
    first_entered = Event()
    release_first = Event()
    writer_calls = []
    logger = _Logger()

    def writer(value):
        writer_calls.append(value)
        if value is first_requested:
            first_entered.set()
            assert release_first.wait(timeout=2)
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        logger=logger,
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(controller.tcp_mirror_setter, first_requested)
        assert first_entered.wait(timeout=2)
        first_correlation = controller._tcp_mirror_write_correlation

        assert controller.tcp_mirror_setter(second_requested) is False
        assert controller._tcp_mirror_write_correlation is first_correlation
        assert writer_calls == [first_requested]
        assert logger.messages.count(
            (
                "error",
                "trigger resource verify-write failed: "
                "tcp-mirror/returned False",
            )
        ) == 1

        release_first.set()
        assert first.result(timeout=2) is True

    assert mirror["value"] is first_requested
    assert controller._tcp_mirror_write_correlation is None
    assert controller._tcp_mirror_write_inflight == 0


def test_tcp_mirror_teardown_precedes_matching_self_admission(qapp):
    owner, lifecycle, controller, _model, _view = _composed_tcp_controller(
        qapp
    )
    requested = object()

    def writer(value):
        with controller._lifecycle_lock:
            controller._lifecycle_state = "DISCONNECTING"
        return lifecycle.write_tcp_mirror_identity(value)

    controller._tcp_mirror_writer = writer
    try:
        assert controller.tcp_mirror_setter(requested) is False
        assert lifecycle.read_tcp_mirror_identity() is None
        assert controller._tcp_mirror_write_correlation is None
        assert controller._tcp_mirror_write_inflight == 0
    finally:
        _retire_composed_tcp_owner(lifecycle)
        owner.deleteLater()


def test_tcp_mirror_external_transition_is_not_self_acknowledged(qapp):
    owner, lifecycle, controller, _model, _view = _composed_tcp_controller(
        qapp
    )
    requested = object()
    external = object()

    def writer(value):
        assert _CANONICAL_TCP_MIRROR_STATE.write(external) is True
        return lifecycle.write_tcp_mirror_identity(value)

    controller._tcp_mirror_writer = writer
    try:
        initial_epoch = controller._resource_identity_epoch

        assert controller.tcp_mirror_setter(requested) is False

        assert lifecycle.read_tcp_mirror_identity() is requested
        assert controller._resource_identity_epoch > initial_epoch + 1
        assert controller._tcp_mirror_write_correlation is None
    finally:
        _retire_composed_tcp_owner(lifecycle)
        owner.deleteLater()


def test_tcp_mirror_same_identity_succeeds_without_admission_callback(qapp):
    owner, lifecycle, controller, _model, _view = _composed_tcp_controller(
        qapp
    )
    candidate = object()
    try:
        assert _CANONICAL_TCP_MIRROR_STATE.write(candidate) is True
        initial_epoch = controller._resource_identity_epoch

        assert controller.tcp_mirror_setter(candidate) is True

        assert lifecycle.read_tcp_mirror_identity() is candidate
        assert controller._resource_identity_epoch == initial_epoch + 1
        assert controller._tcp_mirror_write_correlation is None
    finally:
        _retire_composed_tcp_owner(lifecycle)
        owner.deleteLater()


def test_tcp_mirror_injected_writer_succeeds_without_admission_callback():
    mirror = {"value": None}
    requested = object()
    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=lambda value: mirror.__setitem__("value", value),
    )

    assert controller.tcp_mirror_setter(requested) is True

    assert mirror["value"] is requested
    assert controller._tcp_mirror_write_correlation is None
    assert controller._tcp_mirror_write_inflight == 0


@pytest.mark.parametrize("failure", ["writer", "readback"])
def test_tcp_mirror_failure_cleanup_retires_owned_correlation(failure):
    mirror = {"value": None}
    reads = 0

    def getter():
        nonlocal reads
        reads += 1
        if failure == "readback" and reads == 2:
            raise RuntimeError("readback failed")
        return mirror["value"]

    def writer(value):
        if failure == "writer":
            raise RuntimeError("writer failed")
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=getter,
        tcp_mirror_setter=writer,
    )

    assert controller.tcp_mirror_setter(object()) is False
    assert controller._tcp_mirror_write_inflight == 0
    assert controller._tcp_mirror_write_correlation is None


def test_composed_tcp_admission_rejection_rolls_back_uncommitted_server(qapp):
    created = []

    def factory(**kwargs):
        server = _TcpServer(**kwargs)
        created.append(server)
        return server

    owner, lifecycle, controller, model, view = _composed_tcp_controller(
        qapp, tcp_server_factory=factory
    )
    rejecting_owner = QWidget()
    rejecting_lifecycle = SequenceResourceLifecycleController(
        SequenceResourceLifecycleView(rejecting_owner),
        parent=rejecting_owner,
    )
    rejecting_trigger, *_ = _controller(
        tcp_mirror_getter=rejecting_lifecycle.read_tcp_mirror_identity,
        tcp_mirror_setter=rejecting_lifecycle.write_tcp_mirror_identity,
    )
    rejecting_lifecycle.tcp_resource_port = (
        SequenceTriggerResourceLifecyclePort(rejecting_trigger)
    )
    with rejecting_trigger._lifecycle_lock:
        rejecting_trigger._lifecycle_state = "FINALIZING"

    try:
        assert controller.set_tcp_enabled(
            True, host="127.0.0.1", port=9001
        ) is False
        assert len(created) == 1
        created_server = created[0]
        assert created_server.start_calls == 0
        assert created_server.stop_calls == 1
        assert model.tcp_server is None
        assert lifecycle.read_tcp_mirror_identity() is None
        assert view.tcp_states[-1] is False
    finally:
        controller.stop_tcp()
        _retire_composed_tcp_owner(lifecycle)
        _retire_composed_tcp_owner(rejecting_lifecycle)
        owner.deleteLater()
        rejecting_owner.deleteLater()


def _tcp_payload(timestamp="2026-08-19T10:00:00", label="NG"):
    return json.dumps(
        {
            "RequestType": "102",
            "RequestContent": {"label": label},
            "IsSync": False,
            "Timestamp": timestamp,
        }
    )


def test_tcp_callback_posts_to_owning_controller_instance_channel(qapp):
    mirror = {"server": None}
    controller, model, _view, _clock, emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )

    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001) is True
    server = mirror["server"]
    assert server is model.tcp_server
    assert server.start_calls == 1

    assert server.callback(_tcp_payload(label="OK")) == "ok"
    _drain_events(qapp)

    assert emitted == [StartTestRequested("trigger-0", "tcp", "OK", True, 7)]
    assert model.tcp_last_request_id == "102@2026-08-19T10:00:00"
    assert model.tcp_connected is True


def test_tcp_duplicate_request_id_is_ignored_and_invalid_payload_is_not_posted(qapp):
    mirror = {"server": None}
    controller, _model, _view, _clock, emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    callback = mirror["server"].callback

    assert callback("not-json") == "error, json format error"
    assert callback(_tcp_payload()) == "ok"
    assert callback(_tcp_payload()) == "pass"
    _drain_events(qapp)

    assert len(emitted) == 1


def test_tcp_callback_concurrent_duplicate_is_admitted_once_before_queueing(qapp):
    mirror = {"server": None}
    controller, _model, _view, _clock, emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    callback = mirror["server"].callback
    payload = _tcp_payload(timestamp="same-request")

    with ThreadPoolExecutor(max_workers=12) as executor:
        responses = list(executor.map(lambda _index: callback(payload), range(24)))
    _drain_events(qapp)

    assert responses.count("ok") == 1
    assert responses.count("pass") == 23
    assert len(emitted) == 1


@pytest.mark.parametrize(
    ("payload", "response"),
    [
        ("not-json", "error, json format error"),
        (json.dumps({"RequestType": "102"}), "error, IsSync type error"),
    ],
)
def test_tcp_callback_preserves_malformed_wire_response(payload, response):
    mirror = {"server": None}
    controller, *_ = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)

    assert mirror["server"].callback(payload) == response


def test_tcp_stop_and_clear_are_idempotent_after_external_mirror_clear():
    mirror = {"server": None}
    controller, model, view, _clock, _emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    server = model.tcp_server

    server.stop()
    mirror["server"] = None
    controller.stop_tcp()
    controller.stop_tcp()

    assert server.stop_calls == 2
    assert model.tcp_server is None
    assert model.tcp_enabled is False
    assert model.tcp_connected is False
    assert mirror["server"] is None
    assert view.tcp_states[-1] is False


def test_tcp_package_queued_before_stop_cannot_start_after_server_is_retired(qapp):
    mirror = {"server": None}
    controller, _model, _view, _clock, emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    callback = mirror["server"].callback

    assert callback(_tcp_payload(timestamp="retired")) == "ok"
    controller.stop_tcp()
    _drain_events(qapp)

    assert emitted == []


def test_tcp_callback_weak_reference_does_not_retarget_another_controller(qapp):
    mirror_a = {"server": None}
    emitted_a = []
    controller_a, _model_a, _view_a, _clock_a, _ = _controller(
        emitted=emitted_a,
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror_a["server"],
        tcp_mirror_setter=lambda server: mirror_a.__setitem__("server", server),
    )
    controller_a.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    callback_a = mirror_a["server"].callback
    ref_a = weakref.ref(controller_a)

    mirror_b = {"server": None}
    controller_b, _model_b, _view_b, _clock_b, emitted_b = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror_b["server"],
        tcp_mirror_setter=lambda server: mirror_b.__setitem__("server", server),
    )
    controller_b.set_tcp_enabled(True, host="127.0.0.1", port=9002)

    controller_a.disconnect()
    del controller_a
    gc.collect()
    assert ref_a() is None

    assert callback_a(_tcp_payload(timestamp="old")) == "error, trigger controller unavailable"
    _drain_events(qapp)
    assert emitted_a == []
    assert emitted_b == []


def test_workflow_controller_is_authoritative_for_stale_and_busy_rejection(qapp):
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(workflow_model, bus)
    view = _View(scanner=False)
    trigger = SequenceTriggerController(
        SequenceTriggerModel(),
        view,
        start_publisher=bus.commands.start_test_requested.emit,
        configuration_generation_provider=lambda: 3,
        workflow_active_provider=lambda: False,
        external_mode_available_provider=lambda: True,
        acquisition_mode_provider=lambda: "RECORD_ONLY",
        command_id_factory=(
            lambda counter=iter(range(10)): f"workflow-trigger-{next(counter)}"
        ),
        event_bus=bus,
        logger=_Logger(),
    )

    assert trigger.request_start(source="manual") is True
    workflow_model.configuration_generation = 4
    _drain_events(qapp)
    assert workflow_model.phase is WorkflowPhase.IDLE
    assert view.workflow_rejections == ["stale configuration generation"]

    trigger.configuration_generation_provider = lambda: 4
    assert trigger.request_start(source="manual") is True
    _drain_events(qapp)
    assert workflow_model.phase is WorkflowPhase.PREPARING

    assert trigger.request_start(source="manual") is True
    _drain_events(qapp)
    assert workflow_model.phase is WorkflowPhase.PREPARING
    assert view.workflow_rejections[-1] == "workflow is busy"

    trigger.disconnect()
    workflow.disconnect()


def test_only_one_external_start_can_be_pending_before_workflow_admission(qapp):
    workflow_model = SequenceWorkflowModel(configuration_generation=3)
    bus = SequenceEventBus()
    workflow = SequenceWorkflowController(workflow_model, bus)
    view = _View()
    trigger = SequenceTriggerController(
        SequenceTriggerModel(),
        view,
        start_publisher=bus.commands.start_test_requested.emit,
        configuration_generation_provider=lambda: 3,
        workflow_active_provider=workflow_model.is_workflow_active,
        external_mode_available_provider=lambda: True,
        acquisition_mode_provider=lambda: "RECORD_ONLY",
        regex_rule_loader=lambda: {"name": "default", "pattern": r".*"},
        command_id_factory=(
            lambda counter=iter(range(10)): f"pending-trigger-{next(counter)}"
        ),
        event_bus=bus,
        logger=_Logger(),
    )

    assert trigger.commit_barcode("SN-FIRST", source="hid") is True
    assert trigger.commit_barcode("SN-SECOND", source="hid") is False
    assert view.serial == "SN-FIRST"

    _drain_events(qapp)
    assert workflow_model.phase is WorkflowPhase.PREPARING

    trigger.disconnect()
    workflow.disconnect()


def test_view_wraps_regex_and_tcp_dialogs_without_business_retry_loop():
    calls = []

    class RegexDialog:
        def __init__(self):
            self.destroyed = _Signal()
            self.visible = False

        def open(self):
            calls.append("regex-open")
            self.visible = True

        def isVisible(self):
            return self.visible

        def raise_(self):
            calls.append("regex-raise")

        def activateWindow(self):
            calls.append("regex-activate")

        def exec(self):
            raise AssertionError("regex dialog must not enter a nested loop")

    class TcpDialog:
        def __init__(self, enabled, host, port):
            calls.append((enabled, host, port))
            self.finished = _Signal()
            self.destroyed = _Signal()
            self.visible = False
            self.clicked_ok_flag = False
            self.is_tcp_flag = enabled
            self.ip = host
            self.port = port

        def open(self):
            calls.append("tcp-open")
            self.visible = True

        def isVisible(self):
            return self.visible

        def raise_(self):
            calls.append("tcp-raise")

        def activateWindow(self):
            calls.append("tcp-activate")

        def exec(self):
            raise AssertionError("tcp dialog must not enter a nested loop")

    view = SequenceTriggerView(
        regex_dialog_factory=RegexDialog,
        tcp_dialog_factory=TcpDialog,
    )
    accepted = []
    rejected = []

    assert view.open_regex_dialog() is True
    assert view.open_regex_dialog() is False
    assert view.open_tcp_dialog(
        False,
        "0.0.0.0",
        8000,
        accepted.append,
        lambda: rejected.append(True),
    ) is True
    assert view.open_tcp_dialog(
        False,
        "0.0.0.0",
        8000,
        accepted.append,
        lambda: rejected.append(True),
    ) is False
    dialog = view._tcp_dialog
    dialog.clicked_ok_flag = True
    dialog.is_tcp_flag = True
    dialog.ip = "127.0.0.1"
    dialog.port = 9001
    dialog.finished.emit(0)
    dialog.finished.emit(0)

    assert calls == [
        "regex-open",
        "regex-raise",
        "regex-activate",
        (False, "0.0.0.0", 8000),
        "tcp-open",
        "tcp-raise",
        "tcp-activate",
    ]
    assert accepted == [(True, "127.0.0.1", 9001)]
    assert rejected == []


def test_tcp_dialog_reject_and_destroy_do_not_apply_configuration():
    instances = []

    class TcpDialog:
        def __init__(self, enabled, host, port):
            self.finished = _Signal()
            self.destroyed = _Signal()
            self.clicked_ok_flag = False
            self.is_tcp_flag = enabled
            self.ip = host
            self.port = port
            instances.append(self)

        def open(self):
            return None

    view = SequenceTriggerView(tcp_dialog_factory=TcpDialog)
    accepted = []
    rejected = []

    assert view.open_tcp_dialog(
        False, "127.0.0.1", 8000, accepted.append, lambda: rejected.append(1)
    ) is True
    instances[-1].finished.emit(0)
    assert accepted == []
    assert rejected == [1]

    assert view.open_tcp_dialog(
        False, "127.0.0.1", 8000, accepted.append, lambda: rejected.append(2)
    ) is True
    destroyed = instances[-1]
    destroyed.destroyed.emit()
    destroyed.clicked_ok_flag = True
    destroyed.finished.emit(0)

    assert accepted == []
    assert rejected == [1, 2]


def test_regex_dialog_parent_is_not_misrouted_as_json_path():
    constructor_values = []
    assigned_parents = []

    class RegexDialog:
        def __init__(self, json_file_path=None):
            constructor_values.append(json_file_path)
            self.finished = _Signal()
            self.destroyed = _Signal()

        def setParent(self, parent):
            assigned_parents.append(parent)

        def setWindowModality(self, _modality):
            return None

        def open(self):
            return None

    parent = object()
    view = SequenceTriggerView(
        parent=parent,
        regex_dialog_factory=RegexDialog,
    )

    assert view.open_regex_dialog() is True
    assert constructor_values == [None]
    assert assigned_parents == [parent]


def test_controller_applies_async_tcp_acceptance_once_and_reject_is_noop():
    class AsyncView(_View):
        def open_tcp_dialog(self, enabled, host, port, on_accepted, on_rejected):
            self.dialog_request = (enabled, host, port, on_accepted, on_rejected)
            return True

    writes = []
    mirror = {"server": None}
    view = AsyncView()
    controller, model, *_ = _controller(
        view=view,
        tcp_server_factory=_TcpServer,
        tcp_config_writer=lambda host, port: writes.append((host, port)),
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )

    assert controller.open_tcp_configuration() is True
    accepted = view.dialog_request[3]
    accepted((True, "127.0.0.1", 9100))
    accepted((True, "127.0.0.1", 9200))

    assert writes == [("127.0.0.1", 9100)]
    assert model.tcp_enabled is True
    assert model.tcp_port == 9100
    assert mirror["server"].start_calls == 1

    controller.stop_tcp()
    assert controller.open_tcp_configuration() is True
    view.dialog_request[4]()
    assert writes == [("127.0.0.1", 9100)]
    assert model.tcp_enabled is False


@pytest.mark.parametrize(
    ("source", "expected_label", "expected_skip"),
    [
        ("barcode", "not_labeled", False),
        ("optical", "not_labeled", False),
        ("shortcut", "not_labeled", False),
        ("tcp", "NG", True),
    ],
)
def test_trigger_admission_bridge_starts_legacy_recording_once_and_finishes_idle(
    qapp, source, expected_label, expected_skip
):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=7)
    session_ids = iter(("session-1", "session-2"))
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: next(session_ids),
        session_snapshot_factory=lambda command, configuration: {
            "command_id": command.command_id,
            "source": command.source,
            "label": command.label,
            "skip_sn_regex_validation": command.skip_sn_regex_validation,
            "configuration_generation": command.configuration_generation,
            "configuration": configuration,
        },
    )
    start_commands = []
    bus.commands.start_test_requested.connect(start_commands.append)
    view = _View(scanner=True)
    mirror = {"server": None}
    trigger = SequenceTriggerController(
        SequenceTriggerModel(),
        view,
        start_publisher=bus.commands.start_test_requested.emit,
        barcode_publisher=bus.commands.barcode_committed.emit,
        event_bus=bus,
        configuration_generation_provider=lambda: 7,
        workflow_active_provider=workflow_model.is_workflow_active,
        external_mode_available_provider=lambda: True,
        acquisition_mode_provider=lambda: "RECORD_ONLY",
        regex_rule_loader=lambda: {"name": "default", "pattern": r".*"},
        command_id_factory=(lambda counter=iter(range(10)): f"e2e-{next(counter)}"),
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
        logger=_Logger(),
    )
    legacy_calls = []
    holder = {}

    def legacy_start(admission, terminal):
        legacy_calls.append(admission)
        terminal.recording_completed(
            sample_count=12,
            result_snapshot={
                "record_id": admission.command_id,
                "label": admission.session_snapshot["label"],
            },
        )
        return True

    bridge = LegacyRecordingAdmissionBridge(bus, legacy_start)
    holder["bridge"] = bridge

    if source == "barcode":
        assert trigger.commit_barcode("SN-1234567", source="hid") is True
    elif source == "optical":
        assert trigger.handle_optical_trigger() is True
    elif source == "shortcut":
        assert trigger.handle_shortcut_trigger() is True
    else:
        assert trigger.set_tcp_enabled(
            True, host="127.0.0.1", port=9001
        ) is True
        assert mirror["server"].callback(
            _tcp_payload(timestamp="e2e", label="NG")
        ) == "ok"
    _drain_events(qapp)

    assert len(start_commands) == 1
    assert len(legacy_calls) == 1
    admission = legacy_calls[0]
    assert admission.command_id == start_commands[0].command_id
    assert admission.session_id == "session-1"
    assert admission.session_snapshot["source"] == start_commands[0].source
    assert admission.session_snapshot["label"] == expected_label
    assert admission.session_snapshot["skip_sn_regex_validation"] is expected_skip
    assert admission.session_snapshot["configuration_generation"] == 7
    assert admission.session_snapshot["workflow_generation"] == 1
    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    trigger.disconnect()
    workflow.disconnect()


def test_admission_bridge_rejection_never_calls_legacy_recording(qapp):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=8)
    workflow = SequenceWorkflowController(workflow_model, bus)
    legacy_calls = []
    bridge = LegacyRecordingAdmissionBridge(bus, legacy_calls.append)
    trigger, _model, _view, _clock, _emitted = _controller(
        generation=lambda: 7,
        event_bus=bus,
        emitted=[],
    )
    trigger.start_publisher = bus.commands.start_test_requested.emit

    assert trigger.handle_optical_trigger() is True
    _drain_events(qapp)

    assert legacy_calls == []
    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    trigger.disconnect()
    workflow.disconnect()


def test_admission_bridge_failed_legacy_start_does_not_leave_preparing(qapp):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=7)
    workflow = SequenceWorkflowController(workflow_model, bus)
    bridge = LegacyRecordingAdmissionBridge(
        bus, lambda _admission, _terminal: False
    )
    trigger, _model, _view, _clock, _emitted = _controller(
        generation=lambda: 7,
        event_bus=bus,
        emitted=[],
    )
    trigger.start_publisher = bus.commands.start_test_requested.emit

    assert trigger.handle_optical_trigger() is True
    _drain_events(qapp)

    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    trigger.disconnect()
    workflow.disconnect()


def test_admission_bridge_start_exception_fails_from_preparing_without_started_event(
    qapp,
):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=7)
    workflow = SequenceWorkflowController(workflow_model, bus)
    started = []
    failures = []
    bus.events.recording_started.connect(started.append)
    bus.events.recording_failed.connect(failures.append)

    def fail_start(_admission, _terminal):
        raise RuntimeError("device unavailable")

    bridge = LegacyRecordingAdmissionBridge(bus, fail_start)
    trigger, _model, _view, _clock, _emitted = _controller(
        generation=lambda: 7,
        event_bus=bus,
        emitted=[],
    )
    trigger.start_publisher = bus.commands.start_test_requested.emit

    assert trigger.handle_optical_trigger() is True
    _drain_events(qapp)

    assert started == []
    assert len(failures) == 1
    assert failures[0].reason == "device unavailable"
    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    trigger.disconnect()
    workflow.disconnect()


def test_admission_bridge_allows_next_trigger_after_terminal_completion(qapp):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=7)
    session_ids = iter(("session-1", "session-2"))
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: next(session_ids),
    )
    calls = []
    holder = {}

    def legacy_start(admission, terminal):
        calls.append(admission.session_id)
        terminal.recording_completed(
            sample_count=1,
            result_snapshot={"record_id": admission.session_id},
        )
        return True

    bridge = LegacyRecordingAdmissionBridge(bus, legacy_start)
    holder["bridge"] = bridge
    trigger, _model, _view, _clock, _emitted = _controller(
        generation=lambda: 7,
        event_bus=bus,
        emitted=[],
    )
    trigger.start_publisher = bus.commands.start_test_requested.emit

    assert trigger.handle_optical_trigger() is True
    _drain_events(qapp)
    assert workflow_model.phase is WorkflowPhase.IDLE
    assert trigger.handle_optical_trigger() is True
    _drain_events(qapp)

    assert calls == ["session-1", "session-2"]
    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    trigger.disconnect()
    workflow.disconnect()


def test_sequence_window_admission_snapshot_factory_preserves_replay_contract():
    command = ReplayRequested("replay-command", "replay-button", "record-9")
    configuration = {"name": "frozen-config"}

    snapshot = legacy_recording_session_snapshot(command, configuration)

    assert snapshot == {
        "command_id": "replay-command",
        "source": "replay-button",
        "record_id": "record-9",
        "label": "not_labeled",
        "skip_sn_regex_validation": False,
        "configuration": configuration,
    }


def _admission(index, *, generation=1, replay=False):
    return BeginRecordingRequested(
        command_id=f"command-{index}",
        session_id=f"session-{index}",
        replay=replay,
        session_snapshot={
            "record_id": f"record-{index}",
            "workflow_generation": generation,
        },
    )


def test_admission_bridge_terminal_port_rejects_delayed_previous_session():
    bus = SequenceEventBus()
    terminals = []
    completed = []
    failed = []
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)

    def start(_admission, terminal):
        terminals.append(terminal)
        return True

    bridge = LegacyRecordingAdmissionBridge(
        bus, start, workflow_generation_provider=lambda: 1
    )
    assert bridge.handle_begin_recording(_admission(1)) is True
    assert terminals[0].recording_completed(
        sample_count=2, result_snapshot={"record_id": "record-1"}
    ) is True
    assert bridge.handle_begin_recording(_admission(2)) is True

    assert terminals[0].recording_failed("late A") is False
    assert terminals[1].recording_completed(
        sample_count=3, result_snapshot={"record_id": "record-2"}
    ) is True
    assert [event.session_id for event in completed] == ["session-1", "session-2"]
    assert failed == []


def test_admission_bridge_first_concurrent_terminal_wins_atomically(qapp):
    bus = SequenceEventBus()
    completed = []
    failed = []
    ports = []
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    bridge = LegacyRecordingAdmissionBridge(
        bus,
        lambda _admission, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: 1,
    )
    assert bridge.handle_begin_recording(_admission("atomic")) is True

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda action: (
                    ports[0].recording_completed(
                        sample_count=1, result_snapshot={"result": True}
                    )
                    if action == "complete"
                    else ports[0].recording_failed("failure")
                ),
                ("complete", "fail"),
            )
        )

    _drain_events(qapp)
    assert sorted(results) == [False, True]
    assert len(completed) + len(failed) == 1
    assert bridge.active_admission is None


def test_admission_bridge_rejects_stale_generation_before_state_mutation():
    bus = SequenceEventBus()
    starts = []
    bridge = LegacyRecordingAdmissionBridge(
        bus,
        lambda admission, terminal: starts.append((admission, terminal)) or True,
        workflow_generation_provider=lambda: 2,
    )

    assert bridge.handle_begin_recording(_admission("stale", generation=1)) is False
    assert starts == []
    assert bridge.active_admission is None
    assert bridge.recent_identity_count == 0


def test_admission_bridge_disconnect_invalidates_bound_terminal_port():
    bus = SequenceEventBus()
    ports = []
    bridge = LegacyRecordingAdmissionBridge(
        bus,
        lambda _admission, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: 1,
    )
    assert bridge.handle_begin_recording(_admission("disconnect")) is True

    bridge.disconnect()
    bridge.disconnect()

    assert ports[0].recording_completed(
        sample_count=1, result_snapshot={"result": True}
    ) is False
    assert bridge.active_admission is None


def test_bound_terminal_rejects_stale_workflow_generation_without_mutation():
    bus = SequenceEventBus()
    generation = {"value": 1}
    ports = []
    completed = []
    bus.events.recording_completed.connect(completed.append)
    bridge = LegacyRecordingAdmissionBridge(
        bus,
        lambda _admission, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: generation["value"],
    )
    admission = _admission("generation", generation=1)
    assert bridge.handle_begin_recording(admission) is True

    generation["value"] = 2

    assert ports[0].recording_completed(sample_count=1, result_snapshot={}) is False
    assert bridge.active_admission is admission
    assert completed == []
    bridge.disconnect()


def test_admission_bridge_recent_identity_state_is_bounded_after_soak():
    bus = SequenceEventBus()

    def start(_admission, terminal):
        terminal.recording_completed(sample_count=1, result_snapshot={})
        return True

    bridge = LegacyRecordingAdmissionBridge(
        bus,
        start,
        workflow_generation_provider=lambda: 1,
        recent_identity_limit=128,
    )
    for index in range(10_000):
        assert bridge.handle_begin_recording(_admission(index)) is True

    assert bridge.active_admission is None
    assert bridge.recent_identity_count <= 128


def test_tcp_replacement_invalidates_old_callback_and_queued_package(qapp):
    mirror = {"server": None}
    controller, _model, _view, _clock, emitted = _controller(
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    server_a = mirror["server"]
    assert server_a.callback(_tcp_payload(timestamp="queued-a")) == "ok"

    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9002)
    server_b = mirror["server"]
    assert server_b is not server_a
    assert server_a.callback(_tcp_payload(timestamp="late-a")) == (
        "error, tcp server inactive"
    )
    assert server_b.callback(_tcp_payload(timestamp="active-b")) == "ok"
    _drain_events(qapp)

    assert [message.label for message in emitted] == ["NG"]


def test_controller_disconnect_is_atomic_idempotent_and_rejects_all_entries(qapp):
    mirror = {"server": None}
    view = _View()
    controller, model, _view, _clock, emitted = _controller(
        view=view,
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    callback = mirror["server"].callback
    generation = controller.lifecycle_generation

    controller.disconnect()
    controller.disconnect()

    assert controller.is_active is False
    assert controller.lifecycle_generation == generation + 1
    assert controller.request_start() is False
    assert controller.commit_barcode("SN-1234567") is False
    assert controller.handle_hid_barcode("SN-1234567") is False
    assert controller.handle_optical_trigger() is False
    assert controller.handle_shortcut_trigger() is False
    assert controller.handle_tcp_run_test() is False
    assert controller.open_tcp_configuration() is False
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9002) is False
    assert callback(_tcp_payload(timestamp="after-disconnect")) == (
        "error, trigger controller unavailable"
    )
    _drain_events(qapp)
    assert emitted == []
    assert model.tcp_server is None
    assert view.close_dialog_calls == 1


@pytest.mark.parametrize(
    "first_stop",
    [False, RuntimeError("ordinary"), KeyboardInterrupt(), SystemExit(4)],
)
def test_disconnect_retries_tcp_stop_without_losing_handle_or_lifecycle(first_stop):
    mirror = {"server": None}

    class FlakyServer(_TcpServer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_results = [first_stop, True]

        def stop(self):
            self.stop_calls += 1
            result = self.stop_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    controller, model, *_ = _controller(
        tcp_server_factory=FlakyServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    server = model.tcp_server
    generation = controller.lifecycle_generation

    assert controller.disconnect() is False
    assert controller.is_active is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert controller.lifecycle_generation == generation
    assert model.tcp_server is server
    assert mirror["server"] is server

    assert controller.disconnect() is True
    assert controller.is_active is False
    assert controller.lifecycle_generation == generation + 1
    assert model.tcp_server is None
    assert mirror["server"] is None
    assert server.stop_calls == 2


def test_disconnect_retries_only_failed_manager_stop_step():
    calls = []
    hardware_results = [False, True]

    class Shortcut:
        def stop(self):
            calls.append("shortcut")
            return True

    class Hardware:
        def stop(self):
            calls.append("hardware")
            return hardware_results.pop(0)

    controller, *_ = _controller(
        shortcut_manager=Shortcut(), hardware_manager=Hardware()
    )

    assert controller.disconnect() is False
    assert controller.is_active is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert calls == ["shortcut", "hardware"]
    assert controller.disconnect() is True
    assert controller.is_active is False
    assert calls == ["shortcut", "hardware", "hardware"]


def test_disconnect_enters_transaction_and_rejects_resource_admission():
    mirror = {"server": None}
    calls = []
    hardware_stop_results = [False, True]

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()

        def ensure_config_loaded(self):
            calls.append((self.name, "ensure"))
            return True

        def start(self):
            calls.append((self.name, "start"))
            self.active = True
            self.handle = object()
            return True

        def stop(self):
            calls.append((self.name, "stop"))
            result = (
                hardware_stop_results.pop(0)
                if self.name == "hardware"
                else True
            )
            if result:
                self.active = False
                self.handle = None
            return result

    shortcut = Manager("shortcut")
    hardware = Manager("hardware")
    controller, model, view, *_ = _controller(
        hardware_manager=hardware,
        shortcut_manager=shortcut,
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert controller.is_active is False
    calls_after_disconnect = list(calls)
    assert controller.request_start() is False
    assert controller.handle_hid_barcode("SN-1234567") is False
    assert controller.handle_optical_trigger() is False
    assert controller.handle_shortcut_trigger() is False
    assert controller.handle_tcp_run_test() is False
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9002) is False
    assert controller.open_tcp_configuration() is False
    assert controller.set_scanner_checked(True) is False
    controller.bind_hardware_signals()
    controller.bind_shortcut_signal()
    assert calls == calls_after_disconnect
    assert model.tcp_server is None
    assert mirror["server"] is None
    assert not hasattr(view, "tcp_dialog_callbacks")

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"


def test_disconnect_rechecks_replaced_or_reactivated_resource_identities():
    mirror = {"server": None}
    calls = []
    hardware_results = [False, True]

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()

        def stop(self):
            calls.append((self.name, self.handle))
            result = hardware_results.pop(0) if self.name == "hardware" else True
            if result:
                self.active = False
                self.handle = None
            return result

    first_shortcut = Manager("shortcut-first")
    hardware = Manager("hardware")
    controller, model, *_ = _controller(
        shortcut_manager=first_shortcut,
        hardware_manager=hardware,
        tcp_server_factory=_TcpServer,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)

    assert controller.disconnect() is False
    replacement_shortcut = Manager("shortcut-replacement")
    replacement_server = _TcpServer()
    model.activate_tcp_server(
        replacement_server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="replacement",
    )
    mirror["server"] = replacement_server
    controller.shortcut_manager = replacement_shortcut

    assert controller.disconnect() is True
    assert replacement_shortcut.active is False
    assert replacement_server.stop_calls == 1
    assert model.tcp_server is None
    assert mirror["server"] is None
    assert [name for name, _handle in calls].count("shortcut-first") == 1
    assert [name for name, _handle in calls].count("shortcut-replacement") == 1


def test_disconnect_rechecks_hardware_reactivated_between_attempts():
    calls = []
    shortcut_results = [False, True]

    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()

        def stop(self):
            calls.append((self.name, self.handle))
            result = (
                shortcut_results.pop(0)
                if self.name == "shortcut"
                else True
            )
            if result:
                self.active = False
                self.handle = None
            return result

    shortcut = Manager("shortcut")
    hardware = Manager("hardware")
    controller, *_ = _controller(
        shortcut_manager=shortcut,
        hardware_manager=hardware,
    )

    assert controller.disconnect() is False
    hardware.active = True
    hardware.handle = object()

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"
    assert [name for name, _handle in calls].count("shortcut") == 2
    assert [name for name, _handle in calls].count("hardware") == 2


@pytest.mark.parametrize("resource", ["shortcut", "hardware"])
def test_disconnect_journal_retries_failed_replaced_manager_instances(resource):
    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return False
            self.active = False
            self.handle = None
            return True

    old = Manager(f"old-{resource}")
    inactive = SimpleNamespace(active=False, handle=None, stop=lambda: True)
    kwargs = {
        "shortcut_manager": old if resource == "shortcut" else inactive,
        "hardware_manager": old if resource == "hardware" else inactive,
    }
    controller, _model, _view, _clock, _emitted = _controller(**kwargs)

    assert controller.disconnect() is False
    replacement = Manager(f"replacement-{resource}")
    setattr(controller, f"{resource}_manager", replacement)

    assert controller.disconnect() is False
    assert old.stop_calls == 2
    assert replacement.stop_calls == 1
    assert controller.lifecycle_state == "DISCONNECTING"

    assert controller.disconnect() is True
    assert replacement.stop_calls == 2
    assert controller.lifecycle_state == "INACTIVE"


def test_disconnect_tcp_journal_retries_failed_old_and_new_servers_by_identity():
    mirror = {"server": None}

    class Server:
        client_address = None

        def __init__(self, **_kwargs):
            self.running = False
            self.stop_calls = 0

        def start(self):
            self.running = True
            return True

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return False
            self.running = False
            return True

    controller, model, _view, _clock, _emitted = _controller(
        tcp_server_factory=Server,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    old = model.tcp_server
    assert controller.disconnect() is False

    replacement = Server()
    replacement.start()
    model.activate_tcp_server(
        replacement,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="replacement",
    )
    mirror["server"] = replacement

    assert controller.disconnect() is False
    assert old.stop_calls == 2
    assert replacement.stop_calls == 1
    assert controller.lifecycle_state == "DISCONNECTING"

    assert controller.disconnect() is True
    assert replacement.stop_calls == 2
    assert model.tcp_server is None
    assert mirror["server"] is None


def test_disconnect_stop_true_but_observably_active_remains_retryable():
    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls > 1:
                self.active = False
                self.handle = None
            return True

    shortcut = Manager()
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=shortcut,
        hardware_manager=SimpleNamespace(
            active=False, handle=None, stop=lambda: True
        ),
    )

    assert controller.disconnect() is False
    assert shortcut.stop_calls == 1
    assert controller.lifecycle_state == "DISCONNECTING"
    assert controller.disconnect() is True
    assert shortcut.stop_calls == 2


def test_disconnect_replacement_pressure_releases_completed_manager_targets():
    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return False
            self.active = False
            self.handle = None
            return True

    managers = [Manager()]
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=managers[0],
        hardware_manager=SimpleNamespace(
            active=False, handle=None, stop=lambda: True
        ),
    )
    assert controller.disconnect() is False

    for _index in range(32):
        replacement = Manager()
        managers.append(replacement)
        controller.shortcut_manager = replacement
        assert controller.disconnect() is False
        assert len(controller._disconnect_resource_journal["shortcut"]) == 1

    assert controller.disconnect() is True
    assert all(manager.stop_calls == 2 for manager in managers)
    assert controller._disconnect_resource_journal["shortcut"] == {}


@pytest.mark.parametrize("resource", ["shortcut", "hardware"])
def test_disconnect_stop_reentrant_manager_replacement_is_journaled(resource):
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
    inactive = SimpleNamespace(active=False, handle=None, stop=lambda: True)
    holder = {}
    old = Manager(
        lambda: setattr(
            holder["controller"], f"{resource}_manager", replacement
        )
    )
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=old if resource == "shortcut" else inactive,
        hardware_manager=old if resource == "hardware" else inactive,
    )
    holder["controller"] = controller

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert old.stop_calls == 1
    assert replacement.stop_calls == 0
    assert list(controller._disconnect_resource_journal[resource].values()) == [
        replacement
    ]

    assert controller.disconnect() is True
    assert replacement.stop_calls == 1
    assert controller.lifecycle_state == "INACTIVE"


def test_disconnect_tcp_stop_reentrant_model_and_mirror_replacements_survive():
    mirror = {"server": None}
    holder = {}

    class Server:
        client_address = None

        def __init__(self, on_stop=None):
            self.running = True
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.running = False
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    new_model = Server()
    new_mirror = Server()
    old_model = Server(
        lambda: holder["model"].activate_tcp_server(
            new_model,
            lifecycle_generation=holder["controller"].lifecycle_generation,
            server_token="new-model",
        )
    )
    old_mirror = Server(lambda: mirror.__setitem__("server", new_mirror))
    old_model_ref = weakref.ref(old_model)
    old_mirror_ref = weakref.ref(old_mirror)
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    holder.update(controller=controller, model=model)
    model.activate_tcp_server(
        old_model,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="old-model",
    )
    mirror["server"] = old_mirror

    assert controller.disconnect() is False
    assert model.tcp_server is new_model
    assert mirror["server"] is new_mirror
    assert new_model.stop_calls == new_mirror.stop_calls == 0
    assert controller.lifecycle_state == "DISCONNECTING"
    del old_model
    del old_mirror
    gc.collect()
    assert old_model_ref() is None
    assert old_mirror_ref() is None

    assert controller.disconnect() is True
    assert new_model.stop_calls == new_mirror.stop_calls == 1
    assert model.tcp_server is None
    assert mirror["server"] is None


def test_disconnect_reentrant_replacement_pressure_keeps_only_one_pending_manager():
    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.on_stop = None

        def stop(self):
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    current = Manager()
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=current,
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    completed_refs = []
    for _index in range(32):
        replacement = Manager()
        current.on_stop = lambda value=replacement: setattr(
            controller, "shortcut_manager", value
        )
        completed_refs.append(weakref.ref(current))
        assert controller.disconnect() is False
        assert len(controller._disconnect_resource_journal["shortcut"]) == 1
        current = replacement
        gc.collect()
        assert completed_refs[-1]() is None

    assert controller.disconnect() is True
    assert controller._disconnect_resource_journal["shortcut"] == {}
    assert all(reference() is None for reference in completed_refs)


def test_disconnect_tcp_mirror_release_reentrant_replacement_is_not_invalidated():
    mirror = {"server": None}
    holder = {}

    class Server:
        client_address = None

        def __init__(self):
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.running = False
            return True

    old = Server()
    replacement = Server()
    release_calls = []

    def set_mirror(server):
        release_calls.append(server)
        if server is None and len(release_calls) == 1:
            mirror["server"] = replacement
            holder["model"].activate_tcp_server(
                replacement,
                lifecycle_generation=holder["controller"].lifecycle_generation,
                server_token="replacement",
            )
            return
        mirror["server"] = server

    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=set_mirror,
    )
    holder.update(controller=controller, model=model)
    model.activate_tcp_server(
        old,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="old",
    )
    mirror["server"] = old

    assert controller.disconnect() is False
    assert model.tcp_server is replacement
    assert mirror["server"] is replacement
    assert replacement.stop_calls == 0
    assert controller.disconnect() is True
    assert replacement.stop_calls == 1


@pytest.mark.parametrize(
    "release_error",
    [RuntimeError("mirror release"), KeyboardInterrupt(), SystemExit(12)],
)
def test_disconnect_tcp_mirror_release_error_journals_installed_replacement(
    release_error,
):
    mirror = {"server": None}
    holder = {}

    class Server:
        client_address = None

        def __init__(self):
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.running = False
            return True

    old = Server()
    replacement = Server()
    release_calls = []

    def set_mirror(server):
        release_calls.append(server)
        if server is None and len(release_calls) == 1:
            mirror["server"] = replacement
            holder["model"].activate_tcp_server(
                replacement,
                lifecycle_generation=holder["controller"].lifecycle_generation,
                server_token="replacement-after-error",
            )
            raise release_error
        mirror["server"] = server

    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=set_mirror,
    )
    holder.update(controller=controller, model=model)
    model.activate_tcp_server(
        old,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="old-before-error",
    )
    mirror["server"] = old

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert list(controller._tcp_stop_journal.values()) == [replacement]
    assert list(controller._tcp_mirror_release_journal.values()) == [old]
    assert replacement.stop_calls == 0

    assert controller.disconnect() is True
    assert replacement.stop_calls == 1
    assert model.tcp_server is None
    assert mirror["server"] is None


@pytest.mark.parametrize(
    "release_error",
    [RuntimeError("model release"), KeyboardInterrupt(), SystemExit(14)],
)
def test_disconnect_tcp_model_release_baseexception_is_retryable(release_error):
    class Server:
        client_address = None

        def __init__(self):
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.running = False
            return True

    class ReleasingModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.release_calls = []

        def invalidate_tcp_server(self):
            self.release_calls.append(True)
            if len(self.release_calls) == 1:
                raise release_error
            return super().invalidate_tcp_server()

    model = ReleasingModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    server = Server()
    model.activate_tcp_server(
        server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="release-retry",
    )
    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert model.tcp_server is server
    assert list(controller._tcp_model_release_journal.values()) == [server]
    assert server.stop_calls == 1

    assert controller.disconnect() is True
    assert model.tcp_server is None
    assert server.stop_calls == 1
    assert len(model.release_calls) == 2


def test_disconnect_manager_same_callback_journals_every_identity_write():
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

    intermediate = Manager()
    replacement = Manager()
    old = Manager(
        lambda: (
            setattr(holder["controller"], "shortcut_manager", intermediate),
            setattr(holder["controller"], "shortcut_manager", replacement),
        )
    )
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=old,
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    holder["controller"] = controller

    assert controller.disconnect() is False
    assert list(controller._disconnect_resource_journal["shortcut"].values()) == [
        intermediate,
        replacement,
    ]
    assert intermediate.stop_calls == replacement.stop_calls == 0

    assert controller.disconnect() is True
    assert intermediate.stop_calls == replacement.stop_calls == 1


def test_disconnect_tcp_same_callback_journals_every_model_and_mirror_write():
    mirror = {"server": None}
    holder = {}

    class Server:
        client_address = None

        def __init__(self, on_stop=None):
            self.running = True
            self.stop_calls = 0
            self.on_stop = on_stop

        def stop(self):
            self.stop_calls += 1
            self.running = False
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    model_intermediate = Server()
    model_replacement = Server()
    mirror_intermediate = Server()
    mirror_replacement = Server()

    def replace_twice():
        controller = holder["controller"]
        model = holder["model"]
        model.activate_tcp_server(
            model_intermediate,
            lifecycle_generation=controller.lifecycle_generation,
            server_token="model-intermediate",
        )
        model.activate_tcp_server(
            model_replacement,
            lifecycle_generation=controller.lifecycle_generation,
            server_token="model-replacement",
        )
        controller.tcp_mirror_setter(mirror_intermediate)
        controller.tcp_mirror_setter(mirror_replacement)

    old = Server(replace_twice)
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    holder.update(controller=controller, model=model)
    model.activate_tcp_server(
        old,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="old",
    )
    mirror["server"] = old

    assert controller.disconnect() is False
    assert list(controller._tcp_stop_journal.values()) == [
        model_intermediate,
        model_replacement,
        mirror_intermediate,
        mirror_replacement,
    ]
    assert all(
        server.stop_calls == 0
        for server in (
            model_intermediate,
            model_replacement,
            mirror_intermediate,
            mirror_replacement,
        )
    )

    assert controller.disconnect() is True
    assert all(
        server.stop_calls == 1
        for server in (
            model_intermediate,
            model_replacement,
            mirror_intermediate,
            mirror_replacement,
        )
    )
    assert model._tcp_server_identity_observer is None


def test_disconnect_final_stability_review_catches_cross_resource_reinstall():
    mirror = {"server": None}
    holder = {}

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

    def reinstall_earlier_resources():
        controller = holder["controller"]
        controller.shortcut_manager = replacement_shortcut
        holder["model"].activate_tcp_server(
            replacement_tcp,
            lifecycle_generation=controller.lifecycle_generation,
            server_token="cross-resource",
        )
        controller.tcp_mirror_setter(replacement_tcp)

    hardware = Resource(on_stop=reinstall_earlier_resources)
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        shortcut_manager=Resource(active=False),
        hardware_manager=hardware,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    holder.update(controller=controller, model=model)

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert replacement_shortcut.stop_calls == replacement_tcp.stop_calls == 0

    assert controller.disconnect() is True
    assert replacement_shortcut.stop_calls == replacement_tcp.stop_calls == 1
    assert controller.lifecycle_state == "INACTIVE"


@pytest.mark.parametrize("release_result", [None, False])
def test_disconnect_mirror_release_requires_confirmed_none(release_result):
    mirror = {"server": None}
    allow_release = {"value": False}

    class Server:
        client_address = None

        def __init__(self):
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.running = False
            return True

    def set_mirror(server):
        if server is None and not allow_release["value"]:
            return release_result
        mirror["server"] = server
        return True

    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=set_mirror,
    )
    server = Server()
    model.activate_tcp_server(
        server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="release-confirmation",
    )
    mirror["server"] = server

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert mirror["server"] is server
    assert model.tcp_server is server
    assert server.stop_calls == 1

    allow_release["value"] = True
    assert controller.disconnect() is True
    assert mirror["server"] is None
    assert model.tcp_server is None
    assert server.stop_calls == 1


def test_disconnect_model_release_requires_confirmed_none():
    class Server:
        client_address = None

        def __init__(self):
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.running = False
            return True

    class NoOpReleaseModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.release_calls = 0

        def invalidate_tcp_server(self):
            self.release_calls += 1
            if self.release_calls == 1:
                return None
            return super().invalidate_tcp_server()

    model = NoOpReleaseModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    server = Server()
    model.activate_tcp_server(
        server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="model-release-confirmation",
    )

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert model.tcp_server is server
    assert server.stop_calls == 1

    assert controller.disconnect() is True
    assert model.tcp_server is None
    assert server.stop_calls == 1


def test_disconnect_false_model_release_retains_exact_handle_until_retry():
    class Server:
        client_address = None

        def __init__(self):
            self.running = True

        def stop(self):
            self.running = False
            return True

    class FalseAfterClearModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.release_calls = 0

        def invalidate_tcp_server(self):
            self.release_calls += 1
            released = super().invalidate_tcp_server()
            if self.release_calls == 1:
                return False
            return released

    model = FalseAfterClearModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    server = Server()
    server_ref = weakref.ref(server)
    model.activate_tcp_server(
        server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="false-model-release",
    )

    assert controller.disconnect() is False
    assert model.tcp_server is None
    del server
    gc.collect()
    assert server_ref() is not None
    assert list(controller._tcp_model_release_journal.values()) == [server_ref()]

    assert controller.disconnect() is True
    gc.collect()
    assert server_ref() is None
    assert controller._tcp_model_release_journal == {}


def test_disconnect_false_mirror_release_retains_exact_handle_until_retry():
    mirror = {"server": None}
    release_calls = []

    class Server:
        client_address = None

        def __init__(self):
            self.running = True

        def stop(self):
            self.running = False
            return True

    def set_mirror(server):
        release_calls.append(server)
        mirror["server"] = server
        if server is None and len(release_calls) == 1:
            return False
        return True

    model_server = Server()
    mirror_server = Server()
    mirror_ref = weakref.ref(mirror_server)
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=set_mirror,
    )
    model.activate_tcp_server(
        model_server,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="false-mirror-release",
    )
    mirror["server"] = mirror_server

    assert controller.disconnect() is False
    assert mirror["server"] is None
    assert model.tcp_server is model_server
    del mirror_server
    gc.collect()
    assert mirror_ref() is not None
    assert list(controller._tcp_mirror_release_journal.values()) == [mirror_ref()]

    assert controller.disconnect() is True
    gc.collect()
    assert mirror_ref() is None
    assert controller._tcp_mirror_release_journal == {}


def test_disconnect_same_callback_replacement_pressure_is_bounded_and_collectable():
    class Manager:
        def __init__(self):
            self.active = True
            self.handle = object()
            self.on_stop = None

        def stop(self):
            self.active = False
            self.handle = None
            if self.on_stop is not None:
                callback, self.on_stop = self.on_stop, None
                callback()
            return True

    current = Manager()
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=current,
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    retired = []
    previous_intermediate_ref = None
    for _index in range(32):
        intermediate = Manager()
        replacement = Manager()
        current.on_stop = lambda a=intermediate, b=replacement: (
            setattr(controller, "shortcut_manager", a),
            setattr(controller, "shortcut_manager", b),
        )
        current_ref = weakref.ref(current)

        assert controller.disconnect() is False
        assert len(controller._disconnect_resource_journal["shortcut"]) == 2

        current = replacement
        retired.append(current_ref)
        if previous_intermediate_ref is not None:
            gc.collect()
            assert previous_intermediate_ref() is None
        previous_intermediate_ref = weakref.ref(intermediate)
        del intermediate
        del replacement
        gc.collect()
        assert current_ref() is None

    assert controller.disconnect() is True
    del current
    gc.collect()
    assert previous_intermediate_ref() is None
    assert all(reference() is None for reference in retired)
    assert controller._disconnect_resource_journal["shortcut"] == {}


def test_controlled_identity_writes_do_not_journal_until_disconnecting():
    class Manager:
        def __init__(self, stop_result=True):
            self.active = True
            self.handle = object()
            self.stop_result = stop_result

        def stop(self):
            if self.stop_result:
                self.active = False
                self.handle = None
            return self.stop_result

    active_replacement = Manager(False)
    controller, model, _view, _clock, _emitted = _controller(
        shortcut_manager=Manager(),
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    controller.shortcut_manager = active_replacement
    active_tcp = Manager(False)
    model.tcp_server = active_tcp
    assert controller._disconnect_resource_journal["shortcut"] == {}
    assert controller._tcp_stop_journal == {}

    assert controller.disconnect() is False
    manager_intermediate = Manager()
    manager_replacement = Manager()
    tcp_intermediate = Manager()
    tcp_replacement = Manager()
    controller.shortcut_manager = manager_intermediate
    controller.shortcut_manager = manager_replacement
    model.tcp_server = tcp_intermediate
    model.tcp_server = tcp_replacement

    assert manager_intermediate in controller._disconnect_resource_journal[
        "shortcut"
    ].values()
    assert manager_replacement in controller._disconnect_resource_journal[
        "shortcut"
    ].values()
    assert tcp_intermediate in controller._tcp_stop_journal.values()
    assert tcp_replacement in controller._tcp_stop_journal.values()


def test_model_identity_outbox_is_atomic_acknowledged_and_capacity_bounded():
    first = object()
    rejected = object()
    model = SequenceTriggerModel(tcp_identity_outbox_limit=1)

    assert model.activate_tcp_server(
        first, lifecycle_generation=3, server_token="first"
    ) is True
    events = model.drain_tcp_identity_outbox()
    assert len(events) == 1
    assert events[0].previous is None
    assert events[0].current is first
    assert model.tcp_enabled is True
    assert model.tcp_running is True
    assert model.tcp_server_token == "first"

    assert model.activate_tcp_server(
        rejected, lifecycle_generation=4, server_token="rejected"
    ) is False
    assert model.tcp_server is first
    assert model.tcp_server_token == "first"
    assert model.tcp_lifecycle_generation == 3
    assert model.drain_tcp_identity_outbox() == events

    assert model.ack_tcp_identity_transition(events[0].sequence) is True
    assert model.ack_tcp_identity_transition(events[0].sequence) is False
    assert model.activate_tcp_server(
        rejected, lifecycle_generation=4, server_token="accepted"
    ) is True
    assert model.tcp_server is rejected


def test_model_direct_identity_assignment_capacity_raises_narrow_before_mutation():
    installed = object()
    rejected = object()
    model = SequenceTriggerModel(tcp_identity_outbox_limit=1)
    model.tcp_server = installed

    with pytest.raises(TcpIdentityOutboxCapacityError):
        model.tcp_server = rejected

    assert model.tcp_server is installed
    assert [
        transition.current for transition in model.drain_tcp_identity_outbox()
    ] == [installed]


def test_tcp_production_enable_contains_model_capacity_rejection_atomically():
    mirror = {"server": None}
    created = []

    class Server(_TcpServer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    model = SequenceTriggerModel(
        tcp_host="old-host", tcp_port=7000, tcp_identity_outbox_limit=0
    )
    controller, _model, view, _clock, _emitted = _controller(
        model=model,
        tcp_server_factory=Server,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )

    assert controller.set_tcp_enabled(
        True, host="new-host", port=9001
    ) is False
    assert len(created) == 1
    assert created[0].start_calls == 0
    assert created[0].stop_calls == 1
    assert model.tcp_server is None
    assert model.tcp_enabled is False
    assert model.tcp_running is False
    assert (model.tcp_host, model.tcp_port) == ("old-host", 7000)
    assert mirror["server"] is None
    assert view.tcp_states == [False]


def test_tcp_model_capacity_rejection_retains_failed_factory_cleanup_for_retry():
    created = []

    class Server:
        client_address = None

        def __init__(self, **_kwargs):
            self.running = True
            self.stop_results = [False, True]
            created.append(self)

        def stop(self):
            result = self.stop_results.pop(0)
            if result:
                self.running = False
            return result

    model = SequenceTriggerModel(tcp_identity_outbox_limit=0)
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        tcp_server_factory=Server,
    )

    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001) is False
    assert list(controller._tcp_stop_journal.values()) == created
    assert controller.stop_tcp() is True
    assert controller._tcp_stop_journal == {}
    assert created[0].running is False


def test_disconnect_observer_unsubscribes_while_weak_guard_rejects_then_prunes():
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    controller_ref = weakref.ref(controller)

    assert model.tcp_identity_observer_count == 1
    assert model.tcp_identity_admission_guard_count == 1
    assert controller.disconnect() is True
    assert model.tcp_identity_observer_count == 0
    assert model.tcp_identity_admission_guard_count == 1
    assert model.activate_tcp_server(
        object(), lifecycle_generation=1, server_token="late"
    ) is False

    del controller
    gc.collect()
    assert controller_ref() is None
    assert model.tcp_identity_admission_guard_count == 0


def test_controller_model_outbox_ack_pressure_is_bounded_and_collectable():
    class Server:
        client_address = None

        def __init__(self):
            self.running = True

        def stop(self):
            self.running = False
            return True

    controller, model, _view, _clock, _emitted = _controller()
    previous = None
    retired = []
    for index in range(32):
        current = Server()
        if previous is not None:
            retired.append(weakref.ref(previous))
        assert model.activate_tcp_server(
            current,
            lifecycle_generation=controller.lifecycle_generation,
            server_token=f"pressure-{index}",
        ) is True
        assert model.drain_tcp_identity_outbox() == ()
        previous = current
        del current
        gc.collect()
        assert all(reference() is None for reference in retired)

    assert model.tcp_identity_observer_count == 1
    assert controller.disconnect() is True
    previous_ref = weakref.ref(previous)
    del previous
    gc.collect()
    assert previous_ref() is None
    assert model.drain_tcp_identity_outbox() == ()


def test_model_observer_token_presence_is_exact_and_failed_unsubscribe_is_nondestructive():
    model = SequenceTriggerModel()

    class Observer:
        def notified(self):
            return None

    observer = Observer()
    first = model.subscribe_tcp_identity_observer(observer.notified)
    second = model.subscribe_tcp_identity_observer(observer.notified)

    assert model.has_tcp_identity_observer(first) is True
    assert model.unsubscribe_tcp_identity_observer(999_999) is False
    assert model.has_tcp_identity_observer(first) is True
    assert model.has_tcp_identity_observer(second) is True
    assert model.unsubscribe_tcp_identity_observer(first) is True
    assert model.has_tcp_identity_observer(first) is False
    assert model.has_tcp_identity_observer(second) is True


def test_disconnect_false_observer_unsubscribe_stays_finalizing_and_retries_exact_token():
    class RetryModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_results = [False, True]
            self.unsubscribe_tokens = []

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_tokens.append(token)
            result = self.unsubscribe_results.pop(0)
            if result is False:
                return False
            return super().unsubscribe_tcp_identity_observer(token)

    model = RetryModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    observer_token = controller._model_identity_observer_token

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "FINALIZING"
    assert controller._model_identity_observer_token == observer_token
    assert model.has_tcp_identity_observer(observer_token) is True
    assert model.activate_tcp_server(
        object(), lifecycle_generation=0, server_token="late-finalizing"
    ) is False

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"
    assert model.has_tcp_identity_observer(observer_token) is False
    assert model.unsubscribe_tokens == [observer_token, observer_token]


def test_disconnect_successful_unsubscribe_is_not_repeated_when_first_verification_raises():
    class VerifyRetryModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_tokens = []
            self.verify_results = [KeyboardInterrupt(), False]

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_tokens.append(token)
            return super().unsubscribe_tcp_identity_observer(token)

        def has_tcp_identity_observer(self, token):
            result = self.verify_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    model = VerifyRetryModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    observer_token = controller._model_identity_observer_token

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "FINALIZING"
    assert controller._model_identity_observer_token == observer_token
    assert model.unsubscribe_tokens == [observer_token]

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"
    assert controller._model_identity_observer_token is None
    assert model.unsubscribe_tokens == [observer_token]


@pytest.mark.parametrize(
    "unsubscribe_error",
    [RuntimeError("unsubscribe"), KeyboardInterrupt(), SystemExit(31)],
)
def test_disconnect_observer_unsubscribe_baseexception_is_retryable(
    unsubscribe_error,
):
    class RetryModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_calls = 0

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_calls += 1
            if self.unsubscribe_calls == 1:
                raise unsubscribe_error
            return super().unsubscribe_tcp_identity_observer(token)

    model = RetryModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "FINALIZING"
    assert model.unsubscribe_calls == 1
    assert controller.disconnect() is True
    assert model.unsubscribe_calls == 2


@pytest.mark.parametrize(
    "close_failure",
    [False, RuntimeError("close"), KeyboardInterrupt(), SystemExit(32)],
)
def test_disconnect_dialog_cleanup_failure_is_finalizing_and_success_steps_do_not_repeat(
    close_failure,
):
    class RetryView(_View):
        def __init__(self):
            super().__init__()
            self.close_results = [close_failure, True]

        def close_dialogs(self):
            self.close_dialog_calls += 1
            result = self.close_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    class CountingModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_calls = 0

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_calls += 1
            return super().unsubscribe_tcp_identity_observer(token)

    view = RetryView()
    model = CountingModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model, view=view
    )

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "FINALIZING"
    assert model.unsubscribe_calls == 1
    assert view.close_dialog_calls == 1

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"
    assert model.unsubscribe_calls == 1
    assert view.close_dialog_calls == 2


def test_disconnect_signal_partial_failure_retries_only_failed_exact_connection():
    class Signal:
        def __init__(self, *results):
            self.results = list(results)
            self.calls = 0

        def disconnect(self, _slot):
            self.calls += 1
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    successful = Signal(True)
    retrying = Signal(False, True)
    controller, _model, _view, _clock, _emitted = _controller()
    controller._connections = [
        (successful, SimpleNamespace(deliver=lambda: None)),
        (retrying, SimpleNamespace(deliver=lambda: None)),
    ]

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "FINALIZING"
    assert successful.calls == 1
    assert retrying.calls == 1
    assert [signal for signal, _guard in controller._connections] == [retrying]

    assert controller.disconnect() is True
    assert successful.calls == 1
    assert retrying.calls == 2


def test_disconnect_observer_removed_despite_false_return_is_verified_absent():
    class RemovedButFalseModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_tokens = []

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_tokens.append(token)
            super().unsubscribe_tcp_identity_observer(token)
            return False

    model = RemovedButFalseModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    observer_token = controller._model_identity_observer_token

    assert controller.disconnect() is True
    assert model.has_tcp_identity_observer(observer_token) is False
    assert model.unsubscribe_tokens == [observer_token]


@pytest.mark.parametrize(
    "unsubscribe_error",
    [RuntimeError("removed"), KeyboardInterrupt(), SystemExit(71)],
)
def test_disconnect_observer_removed_before_baseexception_is_verified_absent(
    unsubscribe_error,
):
    class RemovedThenRaisedModel(SequenceTriggerModel):
        def unsubscribe_tcp_identity_observer(self, token):
            super().unsubscribe_tcp_identity_observer(token)
            raise unsubscribe_error

    model = RemovedThenRaisedModel()
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    observer_token = controller._model_identity_observer_token

    assert controller.disconnect() is True
    assert model.has_tcp_identity_observer(observer_token) is False


def test_disconnect_observer_noop_true_remains_retryable_and_preserves_other_token():
    class NoopThenRemoveModel(SequenceTriggerModel):
        def __init__(self):
            super().__init__()
            self.unsubscribe_calls = 0

        def unsubscribe_tcp_identity_observer(self, token):
            self.unsubscribe_calls += 1
            if self.unsubscribe_calls == 1:
                return True
            return super().unsubscribe_tcp_identity_observer(token)

    class Observer:
        def notified(self):
            return None

    model = NoopThenRemoveModel()
    other = Observer()
    other_token = model.subscribe_tcp_identity_observer(other.notified)
    controller, _model, _view, _clock, _emitted = _controller(model=model)
    controller_token = controller._model_identity_observer_token

    assert controller.disconnect() is False
    assert model.has_tcp_identity_observer(controller_token) is True
    assert model.has_tcp_identity_observer(other_token) is True
    assert controller.disconnect() is True
    assert model.has_tcp_identity_observer(controller_token) is False
    assert model.has_tcp_identity_observer(other_token) is True


def test_disconnect_finalization_reentry_is_serialized_without_recursive_runner():
    class ReentrantView(_View):
        def __init__(self):
            super().__init__()
            self.controller = None
            self.depth = 0
            self.max_depth = 0
            self.nested_results = []

        def close_dialogs(self):
            self.close_dialog_calls += 1
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            try:
                if self.close_dialog_calls == 1:
                    self.nested_results.append(self.controller.disconnect())
                return True
            finally:
                self.depth -= 1

    view = ReentrantView()
    controller, _model, _view, _clock, _emitted = _controller(view=view)
    view.controller = controller

    assert controller.disconnect() is True
    assert view.nested_results == [False]
    assert view.max_depth == 1
    assert controller.lifecycle_state == "INACTIVE"


def test_disconnect_persistent_reentrant_finalization_is_bounded_per_outer_call():
    class ReentrantFailingView(_View):
        def __init__(self):
            super().__init__()
            self.controller = None
            self.depth = 0
            self.max_depth = 0

        def close_dialogs(self):
            self.close_dialog_calls += 1
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            try:
                assert self.controller.disconnect() is False
                return False
            finally:
                self.depth -= 1

    view = ReentrantFailingView()
    controller, _model, _view, _clock, _emitted = _controller(view=view)
    view.controller = controller

    assert controller.disconnect() is False
    assert view.max_depth == 1
    assert view.close_dialog_calls <= 2
    assert controller.lifecycle_state == "FINALIZING"


def test_real_qt_connection_tokens_make_external_disconnect_idempotent(qapp):
    class Hardware(QObject):
        sig_barcode = pyqtSignal(object)
        sig_trigger = pyqtSignal()

        active = False

        def stop(self):
            return True

    class Shortcut(QObject):
        sig_triggered = pyqtSignal()

        active = False

        def stop(self):
            return True

    bus = SequenceEventBus()
    hardware = Hardware()
    shortcut = Shortcut()
    controller, _model, _view, _clock, _emitted = _controller(
        event_bus=bus,
        hardware_manager=hardware,
        shortcut_manager=shortcut,
    )
    controller.bind_hardware_signals()
    controller.bind_shortcut_signal()
    tracked = (
        list(controller._connections)
        + list(controller._hardware_connections)
        + list(controller._shortcut_connections)
        + [controller._debounce_connection, controller._tcp_channel_connection]
    )

    assert all(connection.connection_token is not None for connection in tracked)
    for connection in tracked:
        QObject.disconnect(connection.connection_token)

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"


def test_real_qt_connection_sender_native_destroy_is_absent_success(qapp):
    from PyQt5 import sip

    bus = SequenceEventBus()
    controller, _model, _view, _clock, _emitted = _controller(event_bus=bus)
    commands_sender = bus.commands
    sip.delete(commands_sender)
    assert sip.isdeleted(commands_sender)

    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"


def test_disconnect_manager_journal_capacity_rejects_before_identity_mutation():
    class Manager:
        def __init__(self, name):
            self.name = name
            self.active = True
            self.handle = object()
            self.fail = True

        def stop(self):
            if self.fail:
                return False
            self.active = False
            self.handle = None
            return True

    initial = Manager("initial")
    blocker = Manager("hardware-blocker")
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=initial,
        hardware_manager=blocker,
        resource_journal_limit=4,
    )
    assert controller.disconnect() is False

    admitted = [initial]
    rejected = []
    for index in range(100):
        candidate = Manager(str(index))
        previous = controller.shortcut_manager
        controller.shortcut_manager = candidate
        if controller.shortcut_manager is candidate:
            admitted.append(candidate)
        else:
            rejected.append(candidate)
            assert controller.shortcut_manager is previous
        assert len(controller._disconnect_resource_journal["shortcut"]) <= 4

    assert rejected
    for manager in admitted:
        manager.fail = False
    assert controller.disconnect() is False
    assert controller._disconnect_resource_journal["shortcut"] == {}
    replacement_after_release = Manager("after-release")
    controller.shortcut_manager = replacement_after_release
    assert controller.shortcut_manager is replacement_after_release
    assert list(controller._disconnect_resource_journal["shortcut"].values()) == [
        replacement_after_release
    ]

    replacement_after_release.fail = False
    blocker.fail = False
    assert controller.disconnect() is True


def test_disconnect_tcp_journal_capacity_rejects_model_replacement_atomically():
    class Server:
        def __init__(self):
            self.running = True
            self.fail = True

        def stop(self):
            if self.fail:
                return False
            self.running = False
            return True

    initial = Server()
    class BlockingManager:
        active = True
        handle = object()
        fail = True

        def stop(self):
            if self.fail:
                return False
            self.active = False
            self.handle = None
            return True

    blocker = BlockingManager()
    model = SequenceTriggerModel()
    controller, _model, _view, _clock, _emitted = _controller(
        model=model,
        hardware_manager=blocker,
        tcp_journal_limit=3,
        tcp_mirror_getter=lambda: None,
        tcp_mirror_setter=lambda _value: True,
    )
    assert model.activate_tcp_server(
        initial,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="initial",
    ) is True
    assert controller.disconnect() is False

    admitted = [initial]
    for index in range(100):
        candidate = Server()
        previous = model.tcp_server
        accepted = model.activate_tcp_server(
            candidate,
            lifecycle_generation=controller.lifecycle_generation,
            server_token=f"replacement-{index}",
        )
        if accepted:
            admitted.append(candidate)
        else:
            assert model.tcp_server is previous
        assert len(controller._tcp_stop_journal) <= 3

    assert len(admitted) == 3
    for server in admitted:
        server.fail = False
    assert controller.disconnect() is False
    assert controller._tcp_stop_journal == {}
    replacement_after_release = Server()
    assert model.activate_tcp_server(
        replacement_after_release,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="after-release",
    ) is True
    assert model.tcp_server is replacement_after_release
    assert list(controller._tcp_stop_journal.values()) == [
        replacement_after_release
    ]

    replacement_after_release.fail = False
    blocker.fail = False
    assert controller.disconnect() is True


def test_disconnecting_mirror_write_reserves_capacity_before_calling_writer():
    previous = object()
    first_requested = object()
    second_requested = object()
    mirror = {"value": previous}
    first_entered = Event()
    release_first = Event()
    writer_calls = []

    def writer(value):
        writer_calls.append(value)
        if value is first_requested:
            first_entered.set()
            assert release_first.wait(2)
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            controller._write_tcp_mirror_identity, first_requested
        )
        assert first_entered.wait(2)
        second = executor.submit(
            controller._write_tcp_mirror_identity, second_requested
        )
        second_result = second.result(timeout=1)
        release_first.set()
        first_result = first.result(timeout=2)
        assert second_result[0] is False
        assert writer_calls == [first_requested]
        assert first_result[0] is True

    assert mirror["value"] is first_requested
    assert set(controller._tcp_stop_journal.values()) == {
        previous,
        first_requested,
    }


def test_disconnecting_mirror_reentrant_actual_identity_is_tracked_by_reservation():
    previous = object()
    requested = object()
    reentrant_actual = object()
    mirror = {"value": previous}

    def writer(value):
        mirror["value"] = value
        mirror["value"] = reentrant_actual
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, observed_previous, actual = controller._write_tcp_mirror_identity(
        requested
    )

    assert written is False
    assert observed_previous is previous
    assert actual is reentrant_actual
    assert set(controller._tcp_stop_journal.values()) == {
        previous,
        requested,
        reentrant_actual,
    }
    assert len(controller._tcp_stop_journal) <= controller._tcp_journal_limit


def test_mirror_write_never_waits_for_holder_while_holding_lifecycle_lock():
    holder_lock = Lock()
    holder_held = Event()
    getter_entered = Event()
    owner_lock_completed = Event()
    writer_completed = Event()
    mirror = {"value": None}

    def getter():
        getter_entered.set()
        with holder_lock:
            return mirror["value"]

    def writer(value):
        with holder_lock:
            mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=getter,
        tcp_mirror_setter=writer,
    )
    requested = object()

    def hold_holder_then_take_owner_lock():
        with holder_lock:
            holder_held.set()
            assert getter_entered.wait(1)
            with controller._lifecycle_lock:
                owner_lock_completed.set()

    def write_mirror():
        controller._write_tcp_mirror_identity(requested)
        writer_completed.set()

    holder_thread = Thread(
        target=hold_holder_then_take_owner_lock, daemon=True
    )
    writer_thread = Thread(target=write_mirror, daemon=True)
    holder_thread.start()
    assert holder_held.wait(1)
    writer_thread.start()

    holder_thread.join(0.3)
    writer_thread.join(0.3)
    assert owner_lock_completed.is_set()
    assert writer_completed.is_set()


def test_false_writer_still_tracks_requested_and_unexpected_actual():
    previous = object()
    requested = object()
    actual = object()
    mirror = {"value": previous}

    def writer(_value):
        mirror["value"] = actual
        return False

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, observed_previous, observed_actual = (
        controller._write_tcp_mirror_identity(requested)
    )

    assert written is False
    assert observed_previous is previous
    assert observed_actual is actual
    assert controller._tcp_mirror_write_reservations == {}
    assert set(controller._tcp_stop_journal.values()) == {
        previous,
        requested,
        actual,
    }


def test_exposed_requested_then_actual_rolls_back_without_capacity_leak():
    class Server:
        def __init__(self):
            self.active = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.active = False
            return True

    previous = Server()
    requested = Server()
    actual = Server()
    mirror = {"value": previous}
    writer_calls = []

    def writer(value):
        writer_calls.append(value)
        if value is requested:
            mirror["value"] = requested
            mirror["value"] = actual
        else:
            mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, _previous, observed_actual = controller._write_tcp_mirror_identity(
        requested
    )

    assert written is False
    assert observed_actual is actual
    assert writer_calls == [requested]
    assert mirror["value"] is actual
    assert actual.stop_calls == 0
    assert actual.active is True
    assert controller._tcp_mirror_write_reservations == {}
    assert len(controller._tcp_stop_journal) <= 3
    assert previous in controller._tcp_stop_journal.values()
    assert requested in controller._tcp_stop_journal.values()
    assert actual in controller._tcp_stop_journal.values()


@pytest.mark.parametrize(
    "writer_error",
    [RuntimeError("write"), KeyboardInterrupt(), SystemExit(73)],
)
def test_mirror_writer_exception_after_mutation_settles_and_disconnects(
    writer_error,
):
    class Server:
        def __init__(self):
            self.active = True

        def stop(self):
            self.active = False
            return True

    previous = Server()
    requested = Server()
    actual = Server()
    mirror = {"value": previous}

    def writer(value):
        if value is requested:
            mirror["value"] = actual
            raise writer_error
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, _previous, observed_actual = controller._write_tcp_mirror_identity(
        requested
    )

    assert written is False
    assert observed_actual is actual
    assert controller._tcp_mirror_write_reservations == {}
    assert len(controller._tcp_stop_journal) <= 3
    assert controller.disconnect() is True


def test_mirror_write_epoch_conflict_settles_reservation_for_retry():
    previous = object()
    requested = object()
    mirror = {"value": previous}
    controller_ref = {}

    def writer(value):
        mirror["value"] = value
        controller = controller_ref["controller"]
        with controller._lifecycle_lock:
            controller._resource_identity_epoch += 1
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    controller_ref["controller"] = controller
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, _previous, actual = controller._write_tcp_mirror_identity(requested)

    assert written is False
    assert actual is requested
    assert controller._tcp_mirror_write_reservations == {}
    assert set(controller._tcp_stop_journal.values()) == {previous, requested}


def test_mirror_writer_is_not_called_without_unexpected_identity_capacity():
    previous = object()
    requested = object()
    mirror = {"value": previous}
    writer_calls = []

    def writer(value):
        writer_calls.append(value)
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=2,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    result = controller._write_tcp_mirror_identity(requested)

    assert result == (False, previous, previous)
    assert writer_calls == []
    assert mirror["value"] is previous
    assert controller._tcp_mirror_write_reservations == {}


def test_untrusted_mirror_actuals_remain_exactly_tracked_when_all_stops_fail():
    class Server:
        def __init__(self, name):
            self.name = name
            self.running = True
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            return False

    previous = Server("previous")
    requested = Server("requested")
    actual = Server("actual")
    mirror = {"value": previous}
    writer_calls = []

    def writer(value):
        writer_calls.append(value)
        mirror["value"] = requested
        mirror["value"] = actual
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    written, observed_previous, observed_actual = (
        controller._write_tcp_mirror_identity(requested)
    )

    assert written is False
    assert observed_previous is previous
    assert observed_actual is actual
    assert writer_calls == [requested]
    assert controller._tcp_mirror_write_reservations == {}
    assert set(controller._tcp_stop_journal.values()) == {
        previous,
        requested,
        actual,
    }
    assert controller.disconnect() is False
    assert set(controller._tcp_stop_journal.values()) == {
        previous,
        requested,
        actual,
    }


def test_mirror_reservation_retains_exact_targets_until_settlement_can_commit():
    import gc
    import weakref

    class Server:
        def __init__(self):
            self.running = True

        def stop(self):
            self.running = False
            return True

    previous = Server()
    requested = Server()
    actual = Server()
    blocker = Server()
    requested_ref = weakref.ref(requested)
    actual_ref = weakref.ref(actual)
    mirror = {"value": previous}
    unexpected = [actual]
    controller_ref = {}

    def writer(value):
        if unexpected:
            mirror["value"] = unexpected.pop()
            controller = controller_ref["controller"]
            with controller._lifecycle_lock:
                controller._tcp_stop_journal[id(blocker)] = blocker
            return False
        mirror["value"] = value
        return True

    controller, _model, _view, _clock, _emitted = _controller(
        tcp_mirror_getter=lambda: mirror["value"],
        tcp_mirror_setter=writer,
        tcp_journal_limit=3,
    )
    controller_ref["controller"] = controller
    with controller._lifecycle_lock:
        controller._lifecycle_state = "DISCONNECTING"

    assert controller._write_tcp_mirror_identity(requested)[0] is False
    assert controller._tcp_mirror_write_reservations
    mirror["value"] = None
    del requested
    del actual
    gc.collect()
    assert requested_ref() is not None
    assert actual_ref() is not None

    with controller._lifecycle_lock:
        controller._tcp_stop_journal.pop(id(blocker), None)
    assert controller.disconnect() is True
    assert controller._tcp_mirror_write_reservations == {}


@pytest.mark.parametrize(
    "observer_error",
    [RuntimeError("observer"), KeyboardInterrupt(), SystemExit(23)],
)
def test_model_identity_observer_is_lock_free_reentrant_and_baseexception_safe(
    observer_error,
):
    model = SequenceTriggerModel(tcp_identity_outbox_limit=8)
    first = object()
    reentrant = object()
    calls = []

    class Observer:
        def notified(self):
            calls.append(model.tcp_server)
            if len(calls) == 1:
                assert model.activate_tcp_server(
                    reentrant,
                    lifecycle_generation=8,
                    server_token="reentrant",
                ) is True
            raise observer_error

    observer = Observer()
    token = model.subscribe_tcp_identity_observer(observer.notified)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            model.activate_tcp_server,
            first,
            lifecycle_generation=7,
            server_token="first",
        )
        assert future.result(timeout=2) is True

    assert calls == [first, reentrant]
    assert model.tcp_server is reentrant
    events = model.drain_tcp_identity_outbox()
    assert [event.current for event in events] == [first, reentrant]
    assert model.unsubscribe_tcp_identity_observer(token) is True
    assert model.unsubscribe_tcp_identity_observer(token) is False


def test_model_identity_observer_weak_method_auto_prunes_after_owner_gc():
    model = SequenceTriggerModel()
    calls = []

    class Observer:
        def notified(self):
            calls.append(True)

    observer = Observer()
    observer_ref = weakref.ref(observer)
    model.subscribe_tcp_identity_observer(observer.notified)
    assert model.tcp_identity_observer_count == 1
    del observer
    gc.collect()
    assert observer_ref() is None

    assert model.activate_tcp_server(
        object(), lifecycle_generation=1, server_token="gc"
    ) is True
    assert calls == []
    assert model.tcp_identity_observer_count == 0


def test_controller_model_observer_installation_does_not_prevent_controller_gc():
    model = SequenceTriggerModel()

    def build_controller_ref():
        controller, _model, _view, _clock, _emitted = _controller(model=model)
        assert model.tcp_identity_observer_count == 1
        return weakref.ref(controller)

    controller_ref = build_controller_ref()
    gc.collect()
    assert controller_ref() is None
    assert model.tcp_identity_observer_count == 0


def test_disconnect_epoch_cas_rejects_final_check_gap_identity_write(monkeypatch):
    replacement = SimpleNamespace(
        active=True,
        handle=object(),
        stop_calls=0,
    )

    def stop_replacement():
        replacement.stop_calls += 1
        replacement.active = False
        replacement.handle = None
        return True

    replacement.stop = stop_replacement
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=SimpleNamespace(active=False, stop=lambda: True),
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
    )
    original_review = controller._disconnect_resources_stable

    def write_after_review():
        stable = original_review()
        controller.shortcut_manager = replacement
        return stable

    monkeypatch.setattr(
        controller, "_disconnect_resources_stable", write_after_review
    )

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert replacement.stop_calls == 0
    monkeypatch.setattr(
        controller, "_disconnect_resources_stable", original_review
    )
    assert controller.disconnect() is True
    assert replacement.stop_calls == 1


def test_inactive_trigger_rejects_manager_model_and_mirror_late_admission():
    mirror = {"server": None}
    controller, model, _view, _clock, _emitted = _controller(
        shortcut_manager=SimpleNamespace(active=False, stop=lambda: True),
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.disconnect() is True
    installed_manager = controller.shortcut_manager
    late_manager = SimpleNamespace(active=True, stop=lambda: True)
    late_model = SimpleNamespace(running=True, stop=lambda: True)
    late_mirror = SimpleNamespace(running=True, stop=lambda: True)

    controller.shortcut_manager = late_manager
    assert controller.shortcut_manager is installed_manager
    assert model.activate_tcp_server(
        late_model,
        lifecycle_generation=controller.lifecycle_generation,
        server_token="late",
    ) is False
    assert model.tcp_server is None
    assert controller.tcp_mirror_setter(late_mirror) is False
    assert mirror["server"] is None


class _HostileStateError(BaseException):
    def __str__(self):
        raise RuntimeError("diagnostics must not stringify this error")


@pytest.mark.parametrize(
    "error",
    [RuntimeError("state"), KeyboardInterrupt(), SystemExit(9), _HostileStateError()],
)
def test_disconnect_state_observation_baseexceptions_are_retryable(error):
    class Manager:
        def __init__(self):
            self.raise_state = True
            self.physically_active = True
            self.stop_calls = 0

        @property
        def active(self):
            if self.raise_state:
                raise error
            return self.physically_active

        def stop(self):
            self.stop_calls += 1
            self.physically_active = False
            return True

    manager = Manager()
    logger = _Logger()
    controller, _model, _view, _clock, _emitted = _controller(
        shortcut_manager=manager,
        hardware_manager=SimpleNamespace(active=False, stop=lambda: True),
        logger=logger,
    )

    assert controller.disconnect() is False
    assert controller.lifecycle_state == "DISCONNECTING"
    assert manager in controller._disconnect_resource_journal["shortcut"].values()
    manager.raise_state = False
    assert controller.disconnect() is True
    assert controller.lifecycle_state == "INACTIVE"


def test_tcp_stop_journals_distinct_model_and_mirror_handles_until_both_stop():
    mirror = {"server": None}

    class Server(_TcpServer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_results = [True]

        def stop(self):
            self.stop_calls += 1
            return self.stop_results.pop(0)

    controller, model, *_ = _controller(
        tcp_server_factory=Server,
        tcp_mirror_getter=lambda: mirror["server"],
        tcp_mirror_setter=lambda server: mirror.__setitem__("server", server),
    )
    assert controller.set_tcp_enabled(True, host="127.0.0.1", port=9001)
    model_server = model.tcp_server
    mirror_server = Server()
    mirror_server.stop_results = [False, True]
    mirror["server"] = mirror_server

    assert controller.stop_tcp() is False
    assert model.tcp_server is model_server
    assert mirror["server"] is mirror_server
    assert model_server.stop_calls == 1
    assert mirror_server.stop_calls == 1

    assert controller.stop_tcp() is True
    assert model.tcp_server is None
    assert mirror["server"] is None
    assert model_server.stop_calls == 1
    assert mirror_server.stop_calls == 2


@pytest.mark.parametrize(
    "error", [RuntimeError("remove failed"), KeyboardInterrupt(), SystemExit(3)]
)
def test_production_shortcut_stop_retains_handle_until_remove_succeeds(
    monkeypatch, error
):
    manager = ShortcutTriggerManager()
    manager.logger = _Logger()
    handle = object()
    manager._hotkey_handle = handle
    outcomes = [error, True]

    def remove_hotkey(actual):
        assert actual is handle
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

    monkeypatch.setattr(
        "base.shortcut_trigger_manager.keyboard.remove_hotkey", remove_hotkey
    )

    assert manager.stop() is False
    assert manager._hotkey_handle is handle
    assert manager.stop() is True
    assert manager._hotkey_handle is None


def test_production_shortcut_start_reports_failure_for_reusable_resume(monkeypatch):
    manager = ShortcutTriggerManager()
    manager.logger = _Logger()
    outcomes = [RuntimeError("register failed"), "registered-handle"]

    def add_hotkey(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("base.shortcut_trigger_manager.keyboard.add_hotkey", add_hotkey)

    assert manager.start() is False
    assert manager._hotkey_handle is None
    assert manager.start() is True
    assert manager._hotkey_handle == "registered-handle"


@pytest.mark.parametrize(
    "error", [RuntimeError("remove failed"), KeyboardInterrupt(), SystemExit(3)]
)
def test_production_hardware_stop_retains_hotkey_until_remove_succeeds(
    qapp, monkeypatch, error
):
    manager = UnifiedHardwareManager()
    manager.logger = _Logger()
    handle = object()
    manager.hotkey_registered = True
    manager._hotkey_handle = handle
    outcomes = [error, True]

    def remove_hotkey(actual):
        assert actual is handle
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

    monkeypatch.setattr(
        "base.unified_hid_device_manager.keyboard.remove_hotkey", remove_hotkey
    )

    assert manager.stop() is False
    assert manager.hotkey_registered is True
    assert manager._hotkey_handle is handle
    assert manager.stop() is True
    assert manager.hotkey_registered is False
    assert manager._hotkey_handle is None


@pytest.mark.parametrize(
    "failure", [False, RuntimeError("close"), KeyboardInterrupt(), SystemExit(7)]
)
@pytest.mark.parametrize("container_kind", ["dict", "list", "single"])
def test_hid_close_retains_only_failed_exact_handles_for_retry(
    qapp, failure, container_kind
):
    manager = UnifiedHardwareManager()
    manager.logger = _Logger()

    class Handle:
        def __init__(self, *results):
            self.results = list(results)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    successful = Handle(True)
    failed = Handle(failure, True)
    if container_kind == "dict":
        manager.hid_handles["scanner"] = {
            "successful": successful,
            "failed": failed,
        }
        expected_retained = {"failed": failed}
    elif container_kind == "list":
        manager.hid_handles["scanner"] = [successful, failed]
        expected_retained = [failed]
    else:
        manager.hid_handles["scanner"] = failed
        expected_retained = failed

    assert manager.close_hid_device("scanner") is False
    assert manager.hid_handles["scanner"] == expected_retained
    assert manager.close_hid_device("scanner") is True
    assert "scanner" not in manager.hid_handles
    assert failed.close_calls == 2
    assert successful.close_calls == (0 if container_kind == "single" else 1)


def test_retained_tcp_dialog_accept_callback_is_generation_guarded_after_disconnect():
    class AsyncView(_View):
        def open_tcp_dialog(self, enabled, host, port, on_accepted, on_rejected):
            self.dialog_request = (enabled, host, port, on_accepted, on_rejected)
            return True

    view = AsyncView()
    writes = []
    controller, model, *_ = _controller(
        view=view,
        tcp_config_writer=lambda host, port: writes.append((host, port)),
    )
    assert controller.open_tcp_configuration() is True
    retained_accept = view.dialog_request[3]

    controller.disconnect()
    retained_accept((True, "127.0.0.1", 9100))

    assert writes == []
    assert model.tcp_enabled is False


def test_regex_dialog_remains_window_modal_top_level_when_parented(qapp):
    parent = QWidget()
    dialogs = []

    class RegexDialog(QDialog):
        def __init__(self):
            super().__init__()
            dialogs.append(self)

        def exec(self):
            raise AssertionError("regex dialog must not enter a nested loop")

    view = SequenceTriggerView(parent=parent, regex_dialog_factory=RegexDialog)

    assert view.open_regex_dialog() is True
    dialog = dialogs[0]
    assert dialog.parent() is parent
    assert dialog.isWindow() is True
    assert dialog.windowModality() == Qt.WindowModal
    assert dialog.isVisible() is True
    view.close_dialogs()


def test_trigger_command_identity_tracking_is_bounded_without_growing_fallback_ids():
    controller, model, *_ = _controller(command_id_factory=lambda: "constant")
    identifiers = [controller._allocate_command_id() for _ in range(10_000)]

    assert not hasattr(model, "_issued_command_ids")
    assert model.recent_command_id_count <= model.recent_command_id_limit
    assert max(map(len, identifiers)) < 100


def test_workflow_replay_admission_dispatches_replay_and_finishes_terminal(qapp):
    bus = SequenceEventBus()
    workflow_model = SequenceWorkflowModel(configuration_generation=7)
    workflow = SequenceWorkflowController(
        workflow_model,
        bus,
        session_id_factory=lambda: "replay-session",
        session_snapshot_factory=legacy_recording_session_snapshot,
    )
    calls = []

    def replay_start(admission, terminal):
        calls.append(admission)
        assert admission.replay is True
        assert terminal.recording_completed(
            sample_count=4,
            result_snapshot={"record_id": admission.session_snapshot["record_id"]},
        )
        return True

    bridge = LegacyRecordingAdmissionBridge(
        bus,
        replay_start,
        workflow_generation_provider=lambda: workflow_model.workflow_generation,
    )

    bus.commands.replay_requested.emit(
        ReplayRequested("replay-command", "replay-button", "record-4")
    )
    _drain_events(qapp)

    assert len(calls) == 1
    assert calls[0].session_id == "replay-session"
    assert calls[0].session_snapshot["record_id"] == "record-4"
    assert calls[0].session_snapshot["workflow_generation"] == 1
    assert workflow_model.phase is WorkflowPhase.IDLE

    bridge.disconnect()
    workflow.disconnect()


def test_bound_terminal_completion_then_failure_emits_completion_only():
    bus = SequenceEventBus()
    completed = []
    failed = []
    ports = []
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    bridge = LegacyRecordingAdmissionBridge(
        bus,
        lambda _admission, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: 1,
    )
    assert bridge.handle_begin_recording(_admission("first-wins")) is True

    assert ports[0].recording_completed(sample_count=1, result_snapshot={}) is True
    assert ports[0].recording_failed("late") is False

    assert len(completed) == 1
    assert failed == []


def test_disconnect_removes_owned_hardware_and_shortcut_bindings():
    class QtSignal:
        def __init__(self):
            self.slots = []

        def connect(self, slot, *_args):
            self.slots.append(slot)

        def disconnect(self, slot):
            self.slots.remove(slot)

        def emit(self, *args):
            for slot in tuple(self.slots):
                slot(*args)

    class Hardware:
        def __init__(self):
            self.sig_barcode = QtSignal()
            self.sig_trigger = QtSignal()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    class Shortcut:
        def __init__(self):
            self.sig_triggered = QtSignal()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    hardware = Hardware()
    shortcut = Shortcut()
    controller, *_ = _controller(
        hardware_manager=hardware, shortcut_manager=shortcut
    )
    controller.bind_hardware_signals()
    controller.bind_shortcut_signal()
    assert len(hardware.sig_barcode.slots) == 1
    assert len(hardware.sig_trigger.slots) == 1
    assert len(shortcut.sig_triggered.slots) == 1

    controller.disconnect()
    controller.disconnect()

    assert hardware.sig_barcode.slots == []
    assert hardware.sig_trigger.slots == []
    assert shortcut.sig_triggered.slots == []
    assert hardware.stop_calls == 1
    assert shortcut.stop_calls == 1


def test_real_tcp_dialog_is_async_parented_window_modal_dialog(qapp, monkeypatch):
    parent = QWidget()
    accepted = []
    rejected = []

    def forbidden_exec(_dialog):
        raise AssertionError("TCP dialog must not enter a nested event loop")

    monkeypatch.setattr(TcpConfigDialog, "exec", forbidden_exec)
    view = SequenceTriggerView(parent=parent)

    assert view.open_tcp_dialog(
        False,
        "127.0.0.1",
        9001,
        accepted.append,
        lambda: rejected.append(True),
    ) is True
    dialog = view._tcp_dialog
    assert type(dialog) is TcpConfigDialog
    assert dialog.parent() is parent
    assert dialog.isWindow() is True
    assert dialog.windowType() == Qt.Dialog
    assert dialog.windowModality() == Qt.WindowModal
    assert dialog.isVisible() is True
    assert dialog.windowFlags() & Qt.WindowCloseButtonHint
    assert dialog.windowFlags() & Qt.WindowMinimizeButtonHint

    dialog.clicked_ok_flag = True
    dialog.is_tcp_flag = True
    dialog.ip = "0.0.0.0"
    dialog.port = "9100"
    dialog.accept()
    _drain_events(qapp)

    assert accepted == [(True, "0.0.0.0", "9100")]
    assert rejected == []

    assert view.open_tcp_dialog(
        True,
        "0.0.0.0",
        "9100",
        accepted.append,
        lambda: rejected.append(True),
    ) is True
    view._tcp_dialog.reject()
    _drain_events(qapp)
    assert accepted == [(True, "0.0.0.0", "9100")]
    assert rejected == [True]


def test_real_regex_dialog_repeated_cycles_delete_transient_qobjects(
    qapp, monkeypatch
):
    parent = QWidget()
    baseline = len(parent.findChildren(SnRegexManageDialog))
    monkeypatch.setattr(
        LoadUiConfig,
        "load_sn_regex_rules_from_json",
        lambda *_args, **_kwargs: LoadUiConfig.build_default_sn_regex_rules_payload(),
    )
    view = SequenceTriggerView(parent=parent)

    for accepted in (True, False, True, False):
        assert view.open_regex_dialog() is True
        assert view.open_regex_dialog() is False
        dialog = view._regex_dialog
        assert type(dialog) is SnRegexManageDialog
        assert dialog.testAttribute(Qt.WA_DeleteOnClose) is True
        dialog_ref = weakref.ref(dialog)

        if accepted:
            dialog.accept()
        else:
            dialog.reject()
        assert view._regex_dialog is None

        _flush_deferred_deletes(qapp)
        assert len(parent.findChildren(SnRegexManageDialog)) == baseline
        del dialog
        gc.collect()
        assert dialog_ref() is None


def test_real_tcp_dialog_repeated_ok_cancel_cycles_delete_once(qapp, monkeypatch):
    parent = QWidget()
    baseline = len(parent.findChildren(TcpConfigDialog))
    accepted = []
    rejected = []

    def forbidden_exec(_dialog):
        raise AssertionError("TCP dialog must not enter a nested event loop")

    monkeypatch.setattr(TcpConfigDialog, "exec", forbidden_exec)
    view = SequenceTriggerView(parent=parent)

    for should_accept in (True, False, True, False):
        assert view.open_tcp_dialog(
            False,
            "127.0.0.1",
            9001,
            accepted.append,
            lambda: rejected.append(True),
        ) is True
        assert view.open_tcp_dialog(
            False,
            "127.0.0.1",
            9001,
            accepted.append,
            lambda: rejected.append(True),
        ) is False
        dialog = view._tcp_dialog
        assert type(dialog) is TcpConfigDialog
        assert dialog.testAttribute(Qt.WA_DeleteOnClose) is True
        dialog_ref = weakref.ref(dialog)

        if should_accept:
            dialog.is_tcp_flag = True
            dialog.ip = "0.0.0.0"
            dialog.port = "9100"
            dialog.ok_btn.click()
        else:
            dialog.cancel_btn.click()
        assert view._tcp_dialog is None

        _flush_deferred_deletes(qapp)
        assert len(parent.findChildren(TcpConfigDialog)) == baseline
        del dialog
        gc.collect()
        assert dialog_ref() is None

    assert accepted == [
        (True, "0.0.0.0", "9100"),
        (True, "0.0.0.0", "9100"),
    ]
    assert rejected == [True, True]


def test_controller_disconnect_deletes_real_dialogs_without_applying_result(
    qapp, monkeypatch
):
    parent = QWidget()
    tcp_baseline = len(parent.findChildren(TcpConfigDialog))
    regex_baseline = len(parent.findChildren(SnRegexManageDialog))
    monkeypatch.setattr(
        LoadUiConfig,
        "load_sn_regex_rules_from_json",
        lambda *_args, **_kwargs: LoadUiConfig.build_default_sn_regex_rules_payload(),
    )
    writes = []
    view = SequenceTriggerView(parent=parent)
    controller, model, *_ = _controller(
        view=view,
        tcp_config_writer=lambda host, port: writes.append((host, port)),
    )

    assert view.open_regex_dialog() is True
    assert controller.open_tcp_configuration() is True
    regex_dialog = view._regex_dialog
    tcp_dialog = view._tcp_dialog
    regex_dialog_ref = weakref.ref(regex_dialog)
    tcp_dialog_ref = weakref.ref(tcp_dialog)
    generation = controller.lifecycle_generation

    controller.disconnect()
    controller.disconnect()

    assert controller.lifecycle_generation == generation + 1
    assert controller.is_active is False
    assert view._regex_dialog is None
    assert view._tcp_dialog is None
    assert writes == []
    assert model.tcp_enabled is False
    _flush_deferred_deletes(qapp)
    assert len(parent.findChildren(TcpConfigDialog)) == tcp_baseline
    assert len(parent.findChildren(SnRegexManageDialog)) == regex_baseline
    del regex_dialog
    del tcp_dialog
    gc.collect()
    assert regex_dialog_ref() is None
    assert tcp_dialog_ref() is None


def test_real_mode_rejection_repeated_cycles_delete_transient_qobjects(qapp):
    parent = QWidget()
    baseline = len(parent.findChildren(_AsyncMessageBox))
    view = SequenceTriggerView(parent=parent, message_box=_AsyncMessageBox)

    for mode, display_name in (
        ("IMPORT_AUDIO", "导入音频"),
        ("IMPORT_STIMULUS_AUDIO", "导入激励与音频"),
        ("IMPORT_AUDIO", "导入音频"),
    ):
        view.show_mode_rejection("扫码", mode)
        dialog = view._external_mode_warning_box
        assert isinstance(dialog, QDialog)
        assert dialog.parent() is parent
        assert dialog.isWindow() is True
        assert dialog.windowType() == Qt.Dialog
        assert dialog.windowModality() == Qt.WindowModal
        assert dialog.testAttribute(Qt.WA_DeleteOnClose) is True
        assert dialog.isVisible() is True
        assert dialog.windowTitle() == "提示"
        assert dialog.text() == (
            f"当前工作模式为 {display_name}，不支持扫码启动工作流。\n"
            "仅【仅录制】和【播放录制】模式支持该功能。"
        )
        dialog_ref = weakref.ref(dialog)

        # The current presentation deduplicates repeated rejection while visible.
        view.show_mode_rejection("扫码", mode)
        assert view._external_mode_warning_box is dialog

        dialog.reject()
        assert view._external_mode_warning_box is None
        _flush_deferred_deletes(qapp)
        assert len(parent.findChildren(_AsyncMessageBox)) == baseline
        del dialog
        gc.collect()
        assert dialog_ref() is None


def test_mode_rejection_stale_old_completion_cannot_clear_replacement(qapp):
    parent = QWidget()
    baseline = len(parent.findChildren(_AsyncMessageBox))
    view = SequenceTriggerView(parent=parent, message_box=_AsyncMessageBox)

    view.show_mode_rejection("扫码", "IMPORT_AUDIO")
    old_dialog = view._external_mode_warning_box
    old_dialog_ref = weakref.ref(old_dialog)
    old_dialog.hide()

    view.show_mode_rejection("快捷键", "IMPORT_STIMULUS_AUDIO")
    replacement = view._external_mode_warning_box
    assert replacement is not old_dialog
    assert replacement.isVisible() is True

    # A queued/late completion from the superseded dialog must not clear the new one.
    old_dialog.finished.emit(QDialog.Rejected)
    assert view._external_mode_warning_box is replacement

    replacement.accept()
    assert view._external_mode_warning_box is None
    _flush_deferred_deletes(qapp)
    assert len(parent.findChildren(_AsyncMessageBox)) == baseline
    del old_dialog
    del replacement
    gc.collect()
    assert old_dialog_ref() is None


def test_controller_disconnect_deletes_mode_rejection_without_late_state_mutation(
    qapp,
):
    parent = QWidget()
    baseline = len(parent.findChildren(_AsyncMessageBox))
    view = SequenceTriggerView(parent=parent, message_box=_AsyncMessageBox)
    controller, *_ = _controller(view=view)

    view.show_mode_rejection("扫码", "IMPORT_AUDIO")
    dialog = view._external_mode_warning_box
    dialog_ref = weakref.ref(dialog)

    controller.disconnect()
    controller.disconnect()
    assert controller.is_active is False
    assert view._external_mode_warning_box is None

    # Even if an already-queued finish signal is delivered, teardown stays final.
    dialog.finished.emit(QDialog.Rejected)
    assert view._external_mode_warning_box is None
    _flush_deferred_deletes(qapp)
    assert len(parent.findChildren(_AsyncMessageBox)) == baseline
    del dialog
    gc.collect()
    assert dialog_ref() is None


def _run_native_windows_message_box_lifecycle() -> None:
    from PyQt5 import sip

    from ui.custom_ui_widget.widgets import MessageBox

    class TrackingTriggerView(SequenceTriggerView):
        def __init__(self, *args, **kwargs):
            self.reference_clear_counts = {}
            super().__init__(*args, **kwargs)

        def __setattr__(self, name, value):
            if name == "_external_mode_warning_box" and hasattr(self, name):
                previous = getattr(self, name)
                if value is not None:
                    self.reference_clear_counts[id(value)] = 0
                if previous is not None and value is None:
                    identifier = id(previous)
                    self.reference_clear_counts[identifier] = (
                        self.reference_clear_counts.get(identifier, 0) + 1
                    )
            super().__setattr__(name, value)

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    parent = QWidget()
    baseline = len(parent.findChildren(MessageBox))
    view = TrackingTriggerView(parent=parent)

    def drain(rounds=4):
        for _ in range(rounds):
            app.processEvents()

    def flush_deferred():
        for _ in range(4):
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
        gc.collect()

    def open_warning(source="扫码", mode="IMPORT_AUDIO"):
        view.show_mode_rejection(source, mode)
        dialog = view._external_mode_warning_box
        assert isinstance(dialog, QMessageBox)
        assert dialog.parent() is parent
        assert dialog.isWindow() is True
        assert dialog.windowModality() == Qt.WindowModal
        assert dialog.testAttribute(Qt.WA_DeleteOnClose) is True
        assert dialog.isVisible() is True
        return dialog

    for action in ("ok", "reject", "ok", "reject"):
        dialog = open_warning()
        identifier = id(dialog)
        dialog_ref = weakref.ref(dialog)
        finished = []
        destroyed = []
        dialog.finished.connect(lambda result, sink=finished: sink.append(result))
        dialog.destroyed.connect(lambda *_args, sink=destroyed: sink.append(True))
        if action == "ok":
            button = dialog.button(QMessageBox.Ok)
            assert button is not None
            button.click()
            del button
        else:
            dialog.reject()
        drain()
        assert view._external_mode_warning_box is None
        assert len(finished) == 1
        assert view.reference_clear_counts[identifier] == 1
        flush_deferred()
        assert len(destroyed) == 1
        assert view.reference_clear_counts[identifier] == 1
        assert sip.isdeleted(dialog)
        assert len(parent.findChildren(MessageBox)) == baseline
        del dialog
        gc.collect()
        assert dialog_ref() is None
        print(f"native mode dialog cycle {action}: reclaimed", flush=True)

    old_dialog = open_warning()
    old_identifier = id(old_dialog)
    old_ref = weakref.ref(old_dialog)
    old_dialog.hide()
    assert old_dialog.isVisible() is False
    replacement = open_warning("快捷键", "IMPORT_STIMULUS_AUDIO")
    replacement_identifier = id(replacement)
    replacement_ref = weakref.ref(replacement)
    assert replacement is not old_dialog
    assert view.reference_clear_counts[old_identifier] == 1
    old_dialog.finished.emit(QDialog.Rejected)
    assert view._external_mode_warning_box is replacement
    assert view.reference_clear_counts[old_identifier] == 1
    flush_deferred()
    assert sip.isdeleted(old_dialog)
    assert view._external_mode_warning_box is replacement
    assert view.reference_clear_counts[old_identifier] == 1
    del old_dialog
    gc.collect()
    assert old_ref() is None
    replacement.reject()
    assert view._external_mode_warning_box is None
    flush_deferred()
    assert sip.isdeleted(replacement)
    assert view.reference_clear_counts[replacement_identifier] == 1
    del replacement
    gc.collect()
    assert replacement_ref() is None
    assert len(parent.findChildren(MessageBox)) == baseline
    print("native mode dialog replacement: stale signals ignored", flush=True)

    destroyed_first = open_warning()
    destroyed_identifier = id(destroyed_first)
    destroyed_first_ref = weakref.ref(destroyed_first)
    finished = []
    destroyed = []
    destroyed_first.finished.connect(lambda result: finished.append(result))
    destroyed_first.destroyed.connect(lambda *_args: destroyed.append(True))
    destroyed_first.deleteLater()
    flush_deferred()
    assert finished == []
    assert destroyed == [True]
    assert sip.isdeleted(destroyed_first)
    assert view._external_mode_warning_box is None
    assert view.reference_clear_counts[destroyed_identifier] == 1
    del destroyed_first
    gc.collect()
    assert destroyed_first_ref() is None
    print("native mode dialog destroyed-first: ownership cleared", flush=True)

    controller = SequenceTriggerController(
        SequenceTriggerModel(),
        view,
        start_publisher=lambda _message: None,
        configuration_generation_provider=lambda: 0,
    )
    disconnect_dialog = open_warning()
    disconnect_identifier = id(disconnect_dialog)
    disconnect_ref = weakref.ref(disconnect_dialog)
    disconnect_finished = []
    disconnect_destroyed = []
    disconnect_dialog.finished.connect(
        lambda result: disconnect_finished.append(result)
    )
    disconnect_dialog.destroyed.connect(
        lambda *_args: disconnect_destroyed.append(True)
    )
    controller.disconnect()
    controller.disconnect()
    view.close_dialogs()
    assert controller.is_active is False
    assert view._external_mode_warning_box is None
    assert disconnect_finished == [QDialog.Rejected]
    assert view.reference_clear_counts[disconnect_identifier] == 1
    flush_deferred()
    assert disconnect_destroyed == [True]
    assert sip.isdeleted(disconnect_dialog)
    assert view.reference_clear_counts[disconnect_identifier] == 1
    del disconnect_dialog
    gc.collect()
    assert disconnect_ref() is None
    assert len(parent.findChildren(MessageBox)) == baseline

    controller.deleteLater()
    parent.deleteLater()
    flush_deferred()
    drain()
    assert not any(widget.isVisible() for widget in app.topLevelWidgets())
    print("NATIVE_MODE_DIALOG_LIFECYCLE_OK", flush=True)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="production QMessageBox lifecycle regression requires native Windows Qt",
)
def test_native_windows_message_box_lifecycle_has_no_retained_or_stale_dialogs():
    project_root = Path(__file__).resolve().parents[2]
    child_environment = os.environ.copy()
    child_environment["QT_QPA_PLATFORM"] = "windows"
    child_environment["PYTHONUTF8"] = "1"
    command = (
        "from unit_test.ui.test_sequence_trigger_mvc import "
        "_run_native_windows_message_box_lifecycle; "
        "_run_native_windows_message_box_lifecycle()"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=project_root,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            "native Windows QMessageBox lifecycle subprocess timed out after "
            f"{error.timeout}s\nstdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        )

    assert completed.returncode == 0, (
        "native Windows QMessageBox lifecycle subprocess failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "NATIVE_MODE_DIALOG_LIFECYCLE_OK" in completed.stdout
