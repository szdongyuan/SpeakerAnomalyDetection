"""Small queued CSV events; the service never retains a window or Qt consumer."""
import logging
import threading
import weakref
import time
from dataclasses import replace

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication


class _EventRelay(QObject):
    _event = pyqtSignal(object)

    def __init__(self):
        # No parent: an in-flight service callback keeps this sender alive even
        # after the receiving bridge is destroyed. Qt disconnects its receiver.
        super().__init__()
        self.lock = threading.Lock()
        self.delivery_closed = False
        self.subscription = None

    def enqueue(self, event):
        with self.lock:
            if not self.delivery_closed:
                self._event.emit(event)

    def close(self):
        with self.lock:
            self.delivery_closed = True
            subscription, self.subscription = self.subscription, None
        if subscription is not None:
            subscription.unsubscribe()


class _GuiSubscription:
    def __init__(self, callbacks, key):
        self._callbacks = callbacks
        self._key = key

    def unsubscribe(self):
        self._callbacks.pop(self._key, None)


class RawAudioCsvServiceBridge(QObject):
    closed = pyqtSignal()

    def __init__(self, service, parent=None):
        app = QApplication.instance()
        if app is None or QThread.currentThread() is not app.thread():
            raise RuntimeError("CSV bridge must be created on the QApplication thread")
        super().__init__(parent)
        self.service = service
        self._logger = logging.getLogger(__name__)
        self._callbacks = {}
        self._terminal_generations = {}
        self._service_closed = service.closed.is_set()
        self._relay = _EventRelay()
        self._relay._event.connect(self._deliver, Qt.QueuedConnection)
        self._relay.subscription = service.subscribe(self._relay.enqueue)
        # This callback only touches the independent relay, never a deleted self.
        self.destroyed.connect(self._relay.close)

    @property
    def delivery_closed(self):
        return self._relay.delivery_closed

    @property
    def service_closed(self):
        return self._service_closed

    def subscribe(self, callback, *, owner=None):
        """Subscribe on Qt's thread; pass owner for closures capturing widgets."""
        if self.delivery_closed:
            raise RuntimeError("CSV bridge delivery is closed")
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("CSV consumers must subscribe on the GUI thread")
        bound_owner = getattr(callback, "__self__", None)
        if isinstance(bound_owner, QObject):
            owner = owner if owner is not None else bound_owner
            reference = weakref.WeakMethod(callback)
            def callback(event):
                method = reference()
                if method is not None:
                    method(event)
        key = object()
        self._callbacks[key] = callback
        subscription = _GuiSubscription(self._callbacks, key)
        if owner is not None:
            # PyQt weakly retains bound Python methods. Keep the token alive
            # even when a caller does not retain the returned subscription.
            owner.destroyed.connect(lambda: subscription.unsubscribe())
        return subscription

    def close_delivery(self):
        """Detach UI delivery without changing service ownership or draining it."""
        self._relay.close()
        self._callbacks.clear()

    def begin_shutdown(self):
        self.service.begin_shutdown()

    @pyqtSlot(object)
    def _deliver(self, event):
        if self.delivery_closed:
            return
        if event.timing is not None:
            event = replace(event, qt_delivery_seconds=max(0.0, time.perf_counter() - event.timing.parent_seconds))
        if event.kind == "terminal":
            identity = event.result
            previous = self._terminal_generations.get(identity.task_id, -1)
            if identity.generation <= previous:
                return
            self._terminal_generations[identity.task_id] = identity.generation
        if event.kind == "closed":
            if self._service_closed:
                return
            # Service emits this only after supervisor join. Its threading.Event
            # may be set just after emission; the received event is authoritative.
            self._service_closed = True
            self.closed.emit()
        for key, callback in tuple(self._callbacks.items()):
            if self.delivery_closed or key not in self._callbacks:
                continue
            try:
                callback(event)
            except Exception:
                # External GUI consumer boundary: diagnose arbitrary user-slot
                # failures without unwinding through Qt or affecting the ledger.
                self._logger.exception("CSV UI consumer failed for %s", event.kind)
