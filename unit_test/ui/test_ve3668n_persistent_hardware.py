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
