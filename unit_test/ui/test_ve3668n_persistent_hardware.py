"""GUI-only VE hardware synchronization tests; no native SDK access."""
import sys
import threading
from types import SimpleNamespace

from PyQt5.QtCore import QThread

from ui.recording_service_bridge import RecordingServiceBridge


class ReleaseService:
    def __init__(self, status="released"):
        self.status = status
        self.busy = False
        self.can_start_recording = True
        self._capture_session = None
        self._pending_ve_release = None
        self._ownership_uncertain = False
        self._worker = None
        self._closing = False
        self.closed = threading.Event()
        self.release_calls = []
        self.callback = None

    def release_ve(self, signature, callback):
        self.release_calls.append(signature)
        self.callback = callback
        if self.status == "pending":
            self._pending_ve_release = object()
        return self.status

    def shutdown(self, callback=None):
        self._closing = True
        if callback is not None:
            callback()


class CallbackBeforePendingService(ReleaseService):
    def release_ve(self, signature, callback):
        self.release_calls.append(signature)
        callback("released", ("completed before return",))
        return "pending"


class PrewarmService(ReleaseService):
    def __init__(self, status="accepted"):
        super().__init__()
        self.prewarm_status = status
        self.prewarm_calls = []
        self.prewarm_callback = None

    def prewarm_ve(self, request, callback):
        self.prewarm_calls.append(request)
        self.prewarm_callback = callback
        return self.prewarm_status


class CallbackBeforeAcceptedService(PrewarmService):
    def __init__(self, completion):
        super().__init__()
        self.completion = completion

    def prewarm_ve(self, request, callback):
        self.prewarm_calls.append(request)
        self.prewarm_callback = callback
        callback(self.completion)
        return "accepted"


def _drain_until(qapp, predicate):
    for _ in range(100):
        qapp.processEvents()
        if predicate():
            return
        QThread.msleep(1)
    assert predicate()


def test_bridge_immediate_no_worker_completion_is_queued_on_gui_thread(ui_qapp):
    service = ReleaseService("released")
    bridge = RecordingServiceBridge(service)
    gui_thread = QThread.currentThread()
    completions = []

    status = bridge.release_ve(("vkinging", "machine", (7, 1), 51200),
        lambda result, diagnostics: completions.append(
            (result, diagnostics, QThread.currentThread())))

    assert status == "released"
    assert completions == []
    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [("released", (), gui_thread)]


def test_bridge_pending_release_completes_once_on_gui_thread(ui_qapp):
    service = ReleaseService("pending")
    bridge = RecordingServiceBridge(service)
    gui_thread = QThread.currentThread()
    completions = []

    assert bridge.release_ve(None, lambda *value: completions.append(
        (*value, QThread.currentThread()))) == "pending"
    assert bridge.hardware_busy

    worker = threading.Thread(target=service.callback, args=("released", ("stopped",)))
    worker.start()
    worker.join()
    assert completions == []
    service._pending_ve_release = None
    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [("released", ("stopped",), gui_thread)]
    assert not bridge.hardware_busy


def test_bridge_callback_before_pending_return_cannot_restore_busy_state(ui_qapp):
    service = CallbackBeforePendingService()
    bridge = RecordingServiceBridge(service)
    gui_thread = QThread.currentThread()
    completions = []

    assert bridge.release_ve(None, lambda *value: completions.append(
        (*value, QThread.currentThread()))) == "pending"
    assert completions == []
    _drain_until(ui_qapp, lambda: bool(completions))

    assert completions == [("released", ("completed before return",), gui_thread)]
    assert not bridge.hardware_busy


def test_bridge_release_failure_and_shutdown_are_delivered_asynchronously(ui_qapp):
    service = ReleaseService("pending")
    bridge = RecordingServiceBridge(service)
    completions = []
    bridge.release_ve(None, lambda *value: completions.append(value))
    service._ownership_uncertain = True
    service._worker = SimpleNamespace(retiring=True)
    service._pending_ve_release = None
    service.callback("failed", ("clear failed",))
    assert completions == [] and bridge.hardware_busy
    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [("failed", ("clear failed",))]

    closing = ReleaseService("closing")
    closing_bridge = RecordingServiceBridge(closing)
    rejected = []
    assert closing_bridge.release_ve(None, lambda *value: rejected.append(value)) == "closing"
    assert rejected == []
    _drain_until(ui_qapp, lambda: bool(rejected))
    assert rejected == [("closing", ())]


def test_hardware_busy_excludes_background_results_but_includes_capture_release_and_closing(ui_qapp):
    service = ReleaseService()
    bridge = RecordingServiceBridge(service)
    service.busy = True  # A result session is still finalizing.
    assert not bridge.hardware_busy

    service._capture_session = object()
    assert bridge.hardware_busy
    service._capture_session = None
    service._pending_ve_release = object()
    assert bridge.hardware_busy
    service._pending_ve_release = None
    service._closing = True
    assert bridge.hardware_busy


def test_bridge_prewarm_rejects_non_gui_thread(ui_qapp):
    bridge = RecordingServiceBridge(PrewarmService())
    failures = []

    def invoke():
        try:
            bridge.prewarm_ve(object(), failures.append)
        except Exception as error:
            failures.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert str(failures[0]) == "Recording bridge VE prewarm must start on its GUI thread"


def test_bridge_prewarm_busy_and_closing_are_immediate_without_pending_state(ui_qapp):
    for status in ("busy", "closing"):
        service = PrewarmService(status)
        bridge = RecordingServiceBridge(service)
        completions = []
        request = object()

        assert bridge.prewarm_ve(request, completions.append) == status
        assert service.prewarm_calls == [request]
        assert completions == []
        assert not bridge.hardware_busy


def test_bridge_accepted_prewarm_projects_busy_and_completes_once_on_gui_thread(ui_qapp):
    service = PrewarmService()
    bridge = RecordingServiceBridge(service)
    gui_thread = QThread.currentThread()
    completions = []
    terminal = object()

    assert bridge.prewarm_ve(object(), lambda value: completions.append(
        (value, QThread.currentThread(), bridge.hardware_busy))) == "accepted"
    assert bridge.hardware_busy

    worker = threading.Thread(target=service.prewarm_callback, args=(terminal,))
    worker.start()
    worker.join()
    service.prewarm_callback(terminal)
    assert completions == []
    assert not bridge.hardware_busy

    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [(terminal, gui_thread, False)]


def test_bridge_synchronous_prewarm_completion_cannot_restore_pending_state(ui_qapp):
    terminal = object()
    service = CallbackBeforeAcceptedService(terminal)
    bridge = RecordingServiceBridge(service)
    completions = []

    assert bridge.prewarm_ve(object(), completions.append) == "accepted"
    assert completions == []
    assert not bridge.hardware_busy

    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [terminal]


def test_bridge_busy_discard_does_not_clear_an_already_accepted_prewarm(ui_qapp):
    service = PrewarmService()
    bridge = RecordingServiceBridge(service)

    assert bridge.prewarm_ve("accepted", lambda _value: None) == "accepted"
    accepted_callback = service.prewarm_callback
    service.prewarm_status = "busy"
    assert bridge.prewarm_ve("discarded", lambda _value: None) == "busy"
    assert bridge.hardware_busy

    accepted_callback(object())
    assert not bridge.hardware_busy


def test_bridge_prewarm_and_release_pending_states_clear_independently(ui_qapp):
    service = PrewarmService()
    service.status = "pending"
    bridge = RecordingServiceBridge(service)

    assert bridge.prewarm_ve(object(), lambda _value: None) == "accepted"
    prewarm_callback = service.prewarm_callback
    assert bridge.release_ve(None) == "pending"
    release_callback = service.callback

    prewarm_callback(object())
    assert bridge.hardware_busy
    service._pending_ve_release = None
    release_callback("released", ())
    assert not bridge.hardware_busy


def test_bridge_shutdown_remains_busy_and_late_prewarm_completion_is_exactly_once(ui_qapp):
    service = PrewarmService()
    bridge = RecordingServiceBridge(service)
    completions = []
    terminal = object()

    assert bridge.prewarm_ve(object(), completions.append) == "accepted"
    bridge.shutdown()
    assert bridge.hardware_busy

    service.prewarm_callback(terminal)
    service.prewarm_callback(terminal)
    _drain_until(ui_qapp, lambda: bool(completions))
    assert completions == [terminal]
    assert bridge.hardware_busy


def test_bridge_prewarm_callback_exception_is_contained_and_queue_remains_usable(
        ui_qapp, monkeypatch, caplog):
    service = PrewarmService()
    bridge = RecordingServiceBridge(service)
    terminal = object()
    callback_values = []
    uncaught = []
    queued_after_failure = []

    monkeypatch.setattr(sys, "excepthook", lambda *details: uncaught.append(details))

    def raising_callback(value):
        callback_values.append(value)
        raise RuntimeError("broken prewarm UI consumer")

    assert bridge.prewarm_ve(object(), raising_callback) == "accepted"
    service.prewarm_callback(terminal)
    service.prewarm_callback(terminal)
    assert not bridge.hardware_busy
    bridge._invoke.emit(lambda: queued_after_failure.append(True))

    _drain_until(ui_qapp, lambda: bool(queued_after_failure))
    assert callback_values == [terminal]
    assert uncaught == []
    assert "VE prewarm UI callback failed" in caplog.text

    later = []
    later_terminal = object()
    assert bridge.prewarm_ve(object(), later.append) == "accepted"
    service.prewarm_callback(later_terminal)
    _drain_until(ui_qapp, lambda: bool(later))
    assert later == [later_terminal]
    assert not bridge.hardware_busy
