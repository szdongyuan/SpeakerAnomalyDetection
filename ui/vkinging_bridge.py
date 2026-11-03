"""GUI-thread polling and request-scoped delivery for one shared SDK owner."""
import time
import uuid
import weakref
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from base.vkinging_discovery import DiscoveryResult


class VkingingBridge(QObject):
    state_changed = pyqtSignal(str)
    discovery_changed = pyqtSignal()

    def __init__(self, service, parent=None, *, clock=time.monotonic):
        super().__init__(parent)
        self.service = service
        self._clock = clock
        self._subscriptions = {}
        self._owner_connections = weakref.WeakSet()
        self.devices = ()
        self.diagnostics = ()
        self.discovery_complete = False
        self._discovery_id = None
        self._last_state = service.state
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    @property
    def state(self):
        return self.service.state

    @property
    def busy(self):
        return self.state != 'idle'

    def subscribe(self, owner, request_id, callback):
        """Replace this owner's previous request subscription.

        Failed/cancelled events without release proof remain subscribed until
        ``released`` arrives. Owners can therefore retain attempt/file ownership
        while native cleanup is pending. Completion with release proof, owner
        finish/destruction, explicit unsubscribe, or replacement ends delivery.
        """
        self.unsubscribe(owner)
        reference = weakref.ref(owner)
        if owner not in self._owner_connections:
            self._owner_connections.add(owner)
            owner.destroyed.connect(lambda _=None, ref=reference: self._unsubscribe_reference(ref))
            if hasattr(owner, 'finished'):
                owner.finished.connect(lambda _=None, ref=reference: self._unsubscribe_reference(ref))
        delivery = weakref.WeakMethod(callback) if getattr(callback, '__self__', None) is owner else callback
        self._subscriptions[request_id] = (reference, delivery)

    def _unsubscribe_reference(self, reference):
        for request_id, (owner, _) in list(self._subscriptions.items()):
            if owner is reference or owner() is None or owner() is reference():
                del self._subscriptions[request_id]

    def unsubscribe(self, owner):
        self._unsubscribe_reference(weakref.ref(owner))

    def request_discovery(self, owner, callback=None):
        request_id = uuid.uuid4().hex
        if not self.service.submit_discovery(request_id):
            return None
        self._discovery_id = request_id
        if callback is not None:
            self.subscribe(owner, request_id, callback)
        self._notify_state()
        return request_id

    def start_capture(self, owner, request, callback):
        if not self.service.start_capture(request):
            return False
        # Service only queues descriptors; polling is confined to this GUI thread.
        self.subscribe(owner, request.request_id, callback)
        self._notify_state()
        return True

    def cancel(self, request_id):
        return self.service.cancel(request_id)

    def _notify_state(self):
        state = self.state
        if state != self._last_state:
            self._last_state = state
            self.state_changed.emit(state)

    def poll(self):
        self.service.tick(self._clock())
        for event in self.service.poll_events():
            terminal = event.kind in ('succeeded', 'failed', 'cancelled')
            if event.request_id == self._discovery_id and terminal:
                self._discovery_id = None
                self.discovery_complete = True
                if event.kind == 'succeeded' and isinstance(event.value, DiscoveryResult):
                    self.devices = event.value.devices
                    self.diagnostics = event.value.diagnostics
                else:
                    self.devices = ()
                    self.diagnostics = (event.reason or '设备发现失败',)
                self.discovery_changed.emit()
            subscription = self._subscriptions.get(event.request_id)
            if subscription is not None:
                owner, delivery = subscription
                callback = delivery() if isinstance(delivery, weakref.WeakMethod) else delivery
                completed = event.kind == 'released' or (
                    terminal and (event.resources_released or event.kind == 'succeeded'))
                if completed:
                    del self._subscriptions[event.request_id]
                if owner() is not None and callback is not None:
                    callback(event)
        # A callback can admit another request; event.state may now be obsolete.
        self._notify_state()
        if self.state == 'closed':
            self.timer.stop()

    def close(self):
        self._subscriptions.clear()
        self.service.close()
        self.poll()
