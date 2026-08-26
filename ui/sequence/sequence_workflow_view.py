"""Presentation adapter for canonical Workflow state."""

from __future__ import annotations

from collections.abc import Callable
import logging
from types import BuiltinMethodType, MethodType, ModuleType
from typing import Any
from weakref import WeakMethod, ref

from PyQt5 import sip
from PyQt5.QtCore import QObject, QThread, Qt, pyqtBoundSignal, pyqtSlot

from ui.sequence.sequence_workflow_model import SequenceWorkflowModel


def _qt_object_is_alive(value: QObject) -> bool:
    try:
        return not sip.isdeleted(value)
    except BaseException:
        return False


def _qobject_callback_is_on_view_thread(
    owner: QObject,
    view: QObject,
) -> bool:
    try:
        if not _qobject_is_on_current_thread(view) or not _qt_object_is_alive(
            owner
        ):
            return False
        owner_thread = owner.thread()
        view_thread = view.thread()
        if owner_thread is None or view_thread is None:
            return False
        if not _qt_object_is_alive(owner_thread) or not _qt_object_is_alive(
            view_thread
        ):
            return False
        return owner_thread == view_thread
    except BaseException:
        return False


def _qobject_is_on_current_thread(value: QObject) -> bool:
    try:
        if not _qt_object_is_alive(value):
            return False
        affinity = value.thread()
        current = QThread.currentThread()
        if affinity is None or current is None:
            return False
        if not _qt_object_is_alive(affinity) or not _qt_object_is_alive(current):
            return False
        return current == affinity
    except BaseException:
        return False


class _CallbackReference:
    """Retain ownerless functions while weakly referencing callback owners."""

    def __init__(
        self,
        callback: Callable[[], Any],
        view: QObject,
    ) -> None:
        self._strong_callback: Callable[[], Any] | None = None
        self._weak_method: WeakMethod | None = None
        self._builtin_owner: Any = None
        self._builtin_method_name: str | None = None
        self._qt_owner: Any = None
        self._view: Any = None

        if isinstance(callback, MethodType):
            owner = callback.__self__
            self._configure_bound_owner(owner, view)
            self._weak_method = WeakMethod(callback)
            return
        if isinstance(callback, BuiltinMethodType):
            owner = callback.__self__
            if owner is not None and not isinstance(owner, ModuleType):
                self._configure_bound_owner(owner, view)
                try:
                    self._builtin_owner = ref(owner)
                except TypeError as error:
                    raise TypeError(
                        "bound workflow callback owner must support weak references"
                    ) from error
                name = callback.__name__
                if type(name) is not str:
                    raise TypeError("bound workflow callback name must be plain text")
                self._builtin_method_name = name
                return
        self._strong_callback = callback
        if isinstance(callback, QObject):
            self._configure_bound_owner(callback, view)

    def _configure_bound_owner(self, owner: Any, view: QObject) -> None:
        self._qt_owner = ref(owner) if isinstance(owner, QObject) else None
        self._view = ref(view) if self._qt_owner is not None else None
        if self._qt_owner is not None and not _qobject_callback_is_on_view_thread(
            owner,
            view,
        ):
            raise ValueError("QObject workflow callback owner must share View thread")

    def resolve(self) -> Callable[[], Any] | None:
        if self._qt_owner is not None:
            owner = self._qt_owner()
            view = self._view()
            if (
                owner is None
                or view is None
                or not _qobject_callback_is_on_view_thread(owner, view)
            ):
                return None
        if self._strong_callback is not None:
            return self._strong_callback
        if self._weak_method is not None:
            return self._weak_method()
        if self._builtin_owner is None or self._builtin_method_name is None:
            return None
        owner = self._builtin_owner()
        if owner is None:
            return None
        return getattr(owner, self._builtin_method_name)

    def release(self) -> None:
        self._strong_callback = None
        self._weak_method = None
        self._builtin_owner = None
        self._builtin_method_name = None
        self._qt_owner = None
        self._view = None


class _CallbackReleaseReceiver(QObject):
    """Release callback references without re-entering a destructing View."""

    def __init__(
        self,
        parent: QObject,
        *callback_references: _CallbackReference,
    ) -> None:
        super().__init__(parent)
        self._callback_references = callback_references
        self._released = False

    @pyqtSlot()
    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for callback_reference in self._callback_references:
            callback_reference.release()


class _WorkflowProjectionReceiver(QObject):
    """One connection-owned receiver whose retirement cannot be reversed."""

    def __init__(self, view: "SequenceWorkflowView") -> None:
        super().__init__(view)
        self._view = ref(view)
        self._retired = False

    def retire(self) -> None:
        self._retired = True

    @pyqtSlot(object)
    def deliver(self, message: Any) -> None:
        if self._retired:
            return
        view = self._view()
        if view is not None and _qt_object_is_alive(view):
            view.project_state_changed(message)


class SequenceWorkflowView(QObject):
    """Expose Workflow-owned state without mirroring it in the facade."""

    def __init__(
        self,
        model: SequenceWorkflowModel,
        *,
        refresh_player_button: Callable[[], Any],
        synchronize_shutdown: Callable[[], Any],
        parent: QObject | None = None,
    ) -> None:
        if type(model) is not SequenceWorkflowModel:
            raise TypeError("model must be SequenceWorkflowModel")
        try:
            callbacks_are_callable = callable(refresh_player_button) and callable(
                synchronize_shutdown
            )
        except BaseException as error:
            raise TypeError("workflow presentation ports must be callable") from error
        if not callbacks_are_callable:
            raise TypeError("workflow presentation ports must be callable")
        super().__init__(parent)
        self.model = model
        try:
            self._refresh_player_button = _CallbackReference(
                refresh_player_button,
                self,
            )
            self._synchronize_shutdown = _CallbackReference(
                synchronize_shutdown,
                self,
            )
        except (TypeError, ValueError):
            raise
        except BaseException as error:
            raise TypeError("workflow callback binding failed") from error
        self._callbacks_released = False
        self._state_changed_signal: Any = None
        self._state_changed_receiver: _WorkflowProjectionReceiver | None = None
        self._callback_release_receiver = _CallbackReleaseReceiver(
            self,
            self._refresh_player_button,
            self._synchronize_shutdown,
        )
        self.destroyed.connect(
            self._callback_release_receiver.release,
            Qt.DirectConnection,
        )

    @pyqtSlot()
    def _release_callbacks(self) -> None:
        if self._callbacks_released:
            return
        self._callbacks_released = True
        self._callback_release_receiver.release()
        receiver = self._state_changed_receiver
        if receiver is not None:
            receiver.retire()
        self._state_changed_signal = None
        self._state_changed_receiver = None

    @staticmethod
    def _log_callback_failure(name: str, error: BaseException) -> None:
        try:
            logging.getLogger(__name__).warning(
                "workflow view %s callback failed: %s",
                name,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
        except BaseException:
            return

    @staticmethod
    def _log_thread_rejection(operation: str) -> None:
        try:
            logging.getLogger(__name__).debug(
                "workflow view rejected off-thread or deleted %s",
                operation,
            )
        except BaseException:
            return

    def _invoke(self, name: str, callback_reference: _CallbackReference) -> bool:
        try:
            callback = callback_reference.resolve()
            if callback is None:
                return False
            callback()
        except BaseException as error:
            self._log_callback_failure(name, error)
            return False
        return True

    @property
    def awaiting_label(self) -> bool:
        if not _qt_object_is_alive(self):
            return False
        return self.model.awaiting_label is True

    def connect_state_changed(
        self,
        signal: Any,
        connection_type: Qt.ConnectionType = Qt.QueuedConnection,
    ) -> bool:
        if not _qobject_is_on_current_thread(self):
            self._log_thread_rejection("connect")
            return False
        if self._callbacks_released:
            return False
        self._retire_state_changed_connection()
        receiver = _WorkflowProjectionReceiver(self)
        try:
            signal.connect(receiver.deliver, connection_type)
        except BaseException as error:
            receiver.retire()
            receiver.deleteLater()
            self._log_callback_failure("connect", error)
            return False
        self._state_changed_signal = signal
        self._state_changed_receiver = receiver
        return True

    def disconnect_state_changed(self, signal: Any) -> bool:
        if not _qobject_is_on_current_thread(self):
            self._log_thread_rejection("disconnect")
            return False
        current_signal = self._state_changed_signal
        if current_signal is None:
            return False
        if (
            type(signal) is not pyqtBoundSignal
            or type(current_signal) is not pyqtBoundSignal
        ):
            return False
        try:
            matches = signal == current_signal
        except BaseException as error:
            self._log_callback_failure("signal identity", error)
            return False
        if matches is not True:
            return False
        disconnected = self._retire_state_changed_connection()
        self._release_callbacks()
        return disconnected

    def _retire_state_changed_connection(self) -> bool:
        if not _qobject_is_on_current_thread(self):
            self._log_thread_rejection("retirement")
            return False
        signal = self._state_changed_signal
        receiver = self._state_changed_receiver
        self._state_changed_signal = None
        self._state_changed_receiver = None
        if signal is None or receiver is None:
            return False
        receiver.retire()
        disconnected = True
        try:
            signal.disconnect(receiver.deliver)
        except (TypeError, RuntimeError):
            disconnected = False
        except BaseException as error:
            self._log_callback_failure("disconnect", error)
            disconnected = False
        if _qt_object_is_alive(receiver):
            try:
                receiver.deleteLater()
            except BaseException as error:
                self._log_callback_failure("receiver retirement", error)
        return disconnected

    @pyqtSlot(object)
    def project_state_changed(self, _message: Any = None) -> bool:
        if not _qobject_is_on_current_thread(self):
            return False
        refreshed = self._invoke("refresh", self._refresh_player_button)
        synchronized = self._invoke("shutdown", self._synchronize_shutdown)
        return refreshed and synchronized
