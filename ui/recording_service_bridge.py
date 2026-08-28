"""GUI-thread delivery for the instance-owned recording service.

Only lifecycle values and one cumulative preview cross this Qt boundary. No
audio device, file writer, pipe operation or process wait runs in this adapter.
"""
import logging
import threading

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

from base.recording_service import RecordingCallbacks


class RecordingServiceBridge(QObject):
    shutting_down = pyqtSignal()
    _event = pyqtSignal(object)
    _preview_wakeup = pyqtSignal(str)
    _invoke = pyqtSignal(object)

    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self._callbacks = {}
        self._previews = {}
        self._preview_wakeups = set()
        self._finished = set()
        self._delivered = set()
        self._lock = threading.Lock()
        self._shutdown_requested = False
        self._event.connect(self._deliver, Qt.QueuedConnection)
        self._preview_wakeup.connect(self._deliver_preview, Qt.QueuedConnection)
        self._invoke.connect(self._call, Qt.QueuedConnection)

    def start(self, request, callbacks):
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Recording bridge must be started on its GUI thread")
        self._callbacks[request.request_id] = callbacks
        routed = {}
        for kind in RecordingCallbacks.__dataclass_fields__:
            routed[kind] = lambda session, value=None, kind=kind: self._enqueue(kind, session, value)
        try:
            return self.service.start(request, RecordingCallbacks(**routed))
        except (RuntimeError, ValueError, TypeError):
            self._callbacks.pop(request.request_id, None)
            raise

    def _enqueue(self, kind, session, value):
        key = session.request.request_id
        if kind == "preview":
            with self._lock:
                if key in self._finished:
                    session.release_preview(value.sequence)
                    return
                previous = self._previews.get(key)
                self._previews[key] = (session, value)
                wake = key not in self._preview_wakeups
                self._preview_wakeups.add(key)
            if previous is not None:
                previous[0].release_preview(previous[1].sequence)
            if wake:
                self._preview_wakeup.emit(key)
            return
        if kind in ("result_ready", "accepted", "failed", "cancelled"):
            with self._lock:
                self._finished.add(key)
        self._event.emit((kind, session, value))

    @pyqtSlot(str)
    def _deliver_preview(self, key):
        with self._lock:
            pending = self._previews.pop(key, None)
            self._preview_wakeups.discard(key)
            finished = key in self._finished
        if pending is None:
            return
        session, preview = pending
        try:
            callbacks = self._callbacks.get(key)
            if not finished and callbacks is not None and callbacks.preview is not None:
                callbacks.preview(session, preview)
        finally:
            session.release_preview(preview.sequence)

    @pyqtSlot(object)
    def _deliver(self, event):
        kind, session, value = event
        key = session.request.request_id
        callbacks = self._callbacks.get(key)
        token = (key, kind)
        if callbacks is None or token in self._delivered:
            return
        self._delivered.add(token)
        callback = getattr(callbacks, kind)
        try:
            if callback is not None:
                if kind in ("started", "released"):
                    callback(session)
                else:
                    callback(session, value)
        except Exception as error:
            # The UI extension boundary must never unwind through a Qt slot.
            # Reject provisional delivery; accepted recordings stay successful.
            logging.getLogger(__name__).exception("Recording UI %s failed: %s", kind, error)
            if kind == "result_ready":
                session.reject_result(f"UI result validation failed: {error}")
        finally:
            if kind == "released":
                self._callbacks.pop(key, None)
                self._delivered.difference_update((key, name) for name in RecordingCallbacks.__dataclass_fields__)
                with self._lock:
                    self._finished.discard(key)

    def shutdown(self, callback=None):
        if not self._shutdown_requested:
            self._shutdown_requested = True
            # Invalidate consumers before any already-queued accepted delivery.
            self.shutting_down.emit()
        self.service.shutdown(None if callback is None else lambda: self._invoke.emit(callback))

    @pyqtSlot(object)
    def _call(self, callback):
        callback()


class RecordingProcessorFacade:
    """Compatibility for UI stop/busy checks, never for parent-side capture."""
    def __init__(self, session):
        self.session = session
        self.target_samples = session.request.target_samples
        self._rec_in_sel = session.request.channels
        self.sample_rate = session.request.sample_rate
        self._audio = None

    def set_recorded_audio(self, audio):
        self._audio = audio

    def get_recorded_data(self):
        if self._audio is None:
            raise RuntimeError("Recording result has not been accepted")
        return self._audio.mono

    @property
    def is_recording(self):
        return self.session.state in ("starting", "recording", "finalizing", "delivering")

    def stop_streaming(self):
        self.session.cancel()
