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
        self._delivery_closed = False
        self._shutdown_requested = False
        self._ve_release_pending = False
        self._ve_prewarm_pending = False
        self._ve_prewarm_calls = set()
        self._event.connect(self._deliver, Qt.QueuedConnection)
        self._preview_wakeup.connect(self._deliver_preview, Qt.QueuedConnection)
        self._invoke.connect(self._call, Qt.QueuedConnection)

    @property
    def hardware_busy(self):
        """Whether capture/native ownership work makes hardware edits unsafe.

        Broad ``service.busy`` also includes parent-side result processing.  A
        completed capture must not keep the hardware dialog disabled while its
        WAV result is being read or accepted.
        """
        service = self.service
        lock = getattr(service, "_lock", None)

        def snapshot():
            worker = getattr(service, "_worker", None)
            closed = getattr(service, "closed", None)
            return bool(
                self._shutdown_requested
                or self._ve_release_pending
                or self._ve_prewarm_pending
                or getattr(service, "_closing", False)
                or (closed is not None and closed.is_set())
                or getattr(service, "_capture_session", None) is not None
                or getattr(service, "_pending_ve_release", None) is not None
                or getattr(service, "_ownership_uncertain", False)
                or (worker is not None and getattr(worker, "retiring", False))
            )

        if lock is None:
            return snapshot()
        with lock:
            return snapshot()

    def release_ve(self, required_signature, callback=None):
        """Request native VE synchronization without blocking the Qt thread."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Recording bridge VE release must start on its GUI thread")
        completion_lock = threading.Lock()
        completed = False

        def complete(status, diagnostics=()):
            nonlocal completed
            with completion_lock:
                if completed:
                    return
                completed = True
            self._ve_release_pending = False
            if callback is not None:
                value = (status, tuple(diagnostics))
                self._invoke.emit(lambda value=value: callback(*value))

        # Establish busy before crossing into the service: its supervisor may
        # complete on another thread before ``release_ve`` returns.  Completion
        # is then the only path that clears this call's pending transition, so
        # a returned ``pending`` can never resurrect already-completed state.
        self._ve_release_pending = True
        status = self.service.release_ve(required_signature, complete)
        if status in ("released", "unchanged", "busy", "closing"):
            # Immediate service outcomes still cross the same queued Qt
            # boundary.  The service may also enqueue its callback; ``complete``
            # makes those two legitimate paths exactly-once.
            complete(status, ())
        return status

    def prewarm_ve(self, request, callback):
        """Start one no-file VE prewarm without blocking the Qt thread."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Recording bridge VE prewarm must start on its GUI thread")
        call_token = object()
        completion_lock = threading.Lock()
        completed = False

        def complete(completion):
            nonlocal completed
            with completion_lock:
                if completed:
                    return
                completed = True
            # Clear the bridge-side hardware reservation before queueing user
            # code so its admission snapshot observes the terminal state.
            with self._lock:
                self._ve_prewarm_calls.discard(call_token)
                self._ve_prewarm_pending = bool(self._ve_prewarm_calls)
            if callback is not None:
                def deliver(completion=completion):
                    try:
                        callback(completion)
                    except Exception as error:
                        # The prewarm consumer is a UI extension boundary just
                        # like recording event delivery.  A broken consumer
                        # must not unwind through the queued Qt slot.
                        logging.getLogger(__name__).exception(
                            "VE prewarm UI callback failed: %s", error)

                self._invoke.emit(deliver)

        # Reserve before entering the service because test doubles and shutdown
        # races may complete synchronously before the admission call returns.
        with self._lock:
            self._ve_prewarm_calls.add(call_token)
            self._ve_prewarm_pending = True
        try:
            status = self.service.prewarm_ve(request, complete)
        except Exception:
            with self._lock:
                self._ve_prewarm_calls.discard(call_token)
                self._ve_prewarm_pending = bool(self._ve_prewarm_calls)
            raise
        if status != "accepted":
            with self._lock:
                self._ve_prewarm_calls.discard(call_token)
                self._ve_prewarm_pending = bool(self._ve_prewarm_calls)
        return status

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
        if self._delivery_closed:
            if kind == "preview":
                session.release_preview(value.sequence)
            elif kind == "result_ready":
                session.reject_result("recording consumer was destroyed")
            return
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
