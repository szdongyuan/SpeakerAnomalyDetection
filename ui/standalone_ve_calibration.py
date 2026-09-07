"""In-process VE input calibration for the standalone recorder.

This module deliberately reuses only the passive recording DTOs expected by
``InputCalibration``.  Acquisition remains owned by a ``QThread`` and the
standalone ``VkDaqClient``; no recording service, result reader, WAV, process,
or IPC worker participates.
"""

from __future__ import annotations

import threading
import re
import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets

from base.recording_process_protocol import (
    RecordingCancelled,
    RecordingFailure,
    RecordingRequest,
    RecordingResult,
)
from base.recording_result_reader import RecordingAudio
from base.recording_service import RecordingCallbacks
from base.ve3668n_input import (
    create_input_config,
    normalize_machine_id,
    normalize_model,
    validate_device_snapshot,
    validate_input_config,
)
from consts.ve3668n_consts import VE_MAX_INPUT_CHANNELS


_AIN_ROUTE = re.compile(r"^(?P<device>.+)/AIN(?P<number>[1-9][0-9]*)$")
_LOGGER = logging.getLogger(__name__)


def _validated_current_ain_routes(routes):
    """Validate one current direct-SDK route enumeration without discovery reuse."""
    try:
        raw_routes = tuple(routes)
    except TypeError as error:
        raise ValueError("current device routes must be iterable") from error
    selected_routes = []
    physical_channels = []
    for route in raw_routes:
        if not isinstance(route, str):
            raise ValueError("current device routes must contain only strings")
        match = _AIN_ROUTE.fullmatch(route)
        if match is None:
            if "/AIN" in route:
                raise ValueError(f"malformed current analog input route: {route!r}")
            continue
        physical_channel = int(match.group("number")) - 1
        if physical_channel >= VE_MAX_INPUT_CHANNELS:
            raise ValueError(f"current analog input route is out of range: {route!r}")
        if physical_channel in physical_channels:
            raise ValueError(
                f"duplicate current analog input physical channel: {physical_channel}"
            )
        selected_routes.append(route)
        physical_channels.append(physical_channel)
    if not selected_routes:
        raise ValueError("current device has no usable analog input routes")
    return tuple(selected_routes), tuple(physical_channels)


def _uncertain_native_ownership(error: BaseException) -> bool:
    """Recognize a returned native cleanup failure without hiding its cause."""
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, "ownership_uncertain", False)):
            return True
        operation = str(getattr(current, "operation", "")).lower()
        if any(token in operation for token in ("stoptask", "cleartask", "close")):
            return True
        notes = " ".join(
            str(note).lower() for note in getattr(current, "__notes__", ())
        )
        if "cleanup also failed" in notes and any(
            token in notes for token in ("stoptask", "cleartask", "close")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class Fixed51200VEProfileStore:
    """Read-free, write-free profile adapter for the standalone fixed rate."""

    def load(self, device, calibration_store):
        profile = validate_input_config(create_input_config(51_200))
        fixed_device = validate_device_snapshot(
            {**device, "input_config": profile}
        )
        calibration_store.observe(fixed_device)
        return dict(profile)


class InProcessCalibrationServiceState:
    """The narrow service-like surface consumed by ``InputCalibration``."""

    def __init__(self, bridge):
        self._bridge = bridge

    @property
    def busy(self) -> bool:
        return self._bridge._active_session is not None

    def cancel(self, request_id):
        self._bridge._cancel(request_id)

    def accept_result(self, request_id):
        self._bridge._accept_result(request_id)

    def reject_result(self, request_id, reason="result rejected by caller"):
        self._bridge._reject_result(request_id, reason)

    def release_preview(self, request_id, sequence):
        # Calibration has no preview lease.  The method exists solely for the
        # passive session/facade protocol and intentionally performs no work.
        return None


class InProcessCalibrationSession:
    """One in-process calibration reservation and its provisional result."""

    def __init__(self, service, request, callbacks):
        self.service = service
        self.request = request
        self.callbacks = callbacks
        self.state = "starting"
        self.cancel_requested = False
        self.released = threading.Event()
        self.release_error = None
        self.audio = None
        self.descriptor = None
        self.failure = None
        self.callback_diagnostics = []
        self._cancel_event = threading.Event()
        self._decision_made = False
        self._terminal_delivered = False
        self._thread_finished = False
        self._cleanup_uncertain = False

    def cancel(self):
        self.service.cancel(self.request.request_id)

    def accept_result(self):
        self.service.accept_result(self.request.request_id)

    def reject_result(self, reason="result rejected by caller"):
        self.service.reject_result(self.request.request_id, reason)

    def release_preview(self, sequence):
        self.service.release_preview(self.request.request_id, sequence)


class CalibrationRecorderWorker(QtCore.QObject):
    """Finite single-channel raw-voltage acquisition on a worker ``QThread``."""

    started = QtCore.pyqtSignal()
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object, bool)
    cancelled = QtCore.pyqtSignal(object, object)
    finished = QtCore.pyqtSignal()

    def __init__(self, session, *, client_factory, device_selector=0):
        super().__init__()
        self._session = session
        self._client_factory = client_factory
        self._device_selector = device_selector

    @QtCore.pyqtSlot()
    def run(self):
        request = self._session.request
        self.started.emit()
        try:
            if self._session._cancel_event.is_set():
                self.cancelled.emit(self._cancelled(request), None)
                return
            with self._client_factory() as client:
                devices = tuple(client.list_devices())
                if not devices:
                    raise RuntimeError("current VE device enumeration is empty")
                selected = client.select_device(self._device_selector)
                expected_name = request.device["name"]
                expected_address = request.device["address"]
                enumerated = next(
                    (
                        device
                        for device in devices
                        if getattr(device, "index", None) == self._device_selector
                    ),
                    None,
                )
                if enumerated is None:
                    raise RuntimeError("frozen VE device is missing from current enumeration")
                if (
                    getattr(enumerated, "name", None) != expected_name
                    or getattr(enumerated, "address", None) != expected_address
                    or getattr(selected, "name", expected_name) != expected_name
                    or getattr(selected, "address", expected_address) != expected_address
                ):
                    raise RuntimeError("selected VE device name/address identity changed")
                current_model = normalize_model(
                    client.get_device_attribute(selected, "Model")
                )
                current_machine_id = normalize_machine_id(
                    client.get_device_attribute(selected, "MachineId")
                )
                if current_model != request.device["model"]:
                    raise RuntimeError("selected VE model identity changed")
                if current_machine_id != request.device["machine_id"]:
                    raise RuntimeError("selected VE machine identity changed")
                routes, physical_channels = _validated_current_ain_routes(
                    client.list_channels()
                )
                frozen_physical_channels = tuple(request.device["physical_channels"])
                if physical_channels != frozen_physical_channels:
                    if set(physical_channels) == set(frozen_physical_channels):
                        raise RuntimeError("current analog input route order changed")
                    raise RuntimeError("current analog input route snapshot changed")
                try:
                    route = routes[physical_channels.index(request.channels[0])]
                except ValueError as error:
                    raise RuntimeError(
                        "requested calibration physical channel is no longer routed"
                    ) from error
                result = client.acquire(
                    channels=(route,),
                    mode="iepe_voltage",
                    sample_rate=51_200,
                    samples_per_channel=512_000,
                    min_value=-10.0,
                    max_value=10.0,
                    terminal="single_ended",
                    timeout=12.0,
                    sensitivity=1000.0,
                )
                multi = self._validated_audio_array(
                    result,
                    expected_channel=route,
                )
            if self._session._cancel_event.is_set():
                self.cancelled.emit(self._cancelled(request), None)
                return
            descriptor = RecordingResult(
                request_id=request.request_id,
                purpose=request.purpose,
                path=request.path,
                sample_rate=request.sample_rate,
                channels=request.channels,
                raw_frames=request.target_samples,
                final_frames=request.target_samples,
                metadata_appended=False,
                handles_released=True,
            )
            self.succeeded.emit(
                RecordingAudio(descriptor, multi, multi[:, 0].copy())
            )
        except Exception as error:
            uncertain = _uncertain_native_ownership(error)
            details = [str(error)]
            for note in getattr(error, "__notes__", ()):
                text = str(note)
                if text and text not in details:
                    details.append(text)
            message = "\n".join(details)
            if self._session._cancel_event.is_set():
                self.cancelled.emit(
                    self._cancelled(
                        request,
                        handles_released=not uncertain,
                    ),
                    message if uncertain else None,
                )
                return
            failure = RecordingFailure(
                request_id=request.request_id,
                stage="cleanup" if uncertain else "capture",
                path=request.path,
                message=message,
                handles_released=not uncertain,
            )
            self.failed.emit(failure, uncertain)
        finally:
            self.finished.emit()

    @staticmethod
    def _cancelled(request, *, handles_released=True):
        return RecordingCancelled(
            request_id=request.request_id,
            path=request.path,
            raw_frames=0,
            final_frames=0,
            handles_released=handles_released,
        )

    @staticmethod
    def _validated_audio_array(result, *, expected_channel):
        samples = getattr(result, "samples", None)
        if (
            getattr(result, "mode", None) != "iepe_voltage"
            or tuple(getattr(result, "channels", ())) != (expected_channel,)
            or float(getattr(result, "sample_rate", 0)) != 51_200.0
            or getattr(result, "requested_samples_per_channel", None) != 512_000
            or getattr(result, "actual_samples_per_channel", None) != 512_000
            or not isinstance(samples, (tuple, list))
            or len(samples) != 1
        ):
            raise ValueError("calibration acquisition did not return the exact fixed profile")
        mono = np.asarray(samples[0], dtype=np.float32)
        if mono.ndim != 1 or mono.shape != (512_000,) or not np.isfinite(mono).all():
            raise ValueError("calibration acquisition must be 512000 finite raw volts")
        return mono.copy().reshape(512_000, 1)


class InProcessCalibrationBridge(QtCore.QObject):
    """GUI-thread adapter for the existing input calibration callback contract."""

    shutting_down = QtCore.pyqtSignal()
    busy_changed = QtCore.pyqtSignal(bool)
    lockout_requested = QtCore.pyqtSignal(str)
    operation_failed = QtCore.pyqtSignal(str)
    callback_failed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None, *, client_factory=None, device_selector=0):
        super().__init__(parent)
        if client_factory is None:
            from vkinging_daq import VkDaqClient

            client_factory = VkDaqClient
        self._client_factory = client_factory
        self._device_selector = device_selector
        self._active_session = None
        self._thread = None
        self._worker = None
        self._closed = False
        self._shutdown_emitted = False
        self._request_ids = set()
        self._shutdown_callbacks = []
        self.service = InProcessCalibrationServiceState(self)

    def _require_gui_thread(self):
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("calibration bridge operations must run on the GUI thread")

    def start(self, request, callbacks=None):
        self._require_gui_thread()
        if not isinstance(request, RecordingRequest):
            raise TypeError("start requires a RecordingRequest")
        if self._closed:
            raise RuntimeError("calibration bridge is shutting down")
        if self.service.busy:
            raise RuntimeError("calibration bridge is busy")
        self._validate_request(request)
        if request.request_id in self._request_ids:
            raise ValueError("request_id must be unique for this bridge")
        callbacks = callbacks or RecordingCallbacks()
        if not isinstance(callbacks, RecordingCallbacks):
            raise TypeError("callbacks must be RecordingCallbacks")

        session = InProcessCalibrationSession(self.service, request, callbacks)
        thread = QtCore.QThread(self)
        worker = CalibrationRecorderWorker(
            session,
            client_factory=self._client_factory,
            device_selector=self._device_selector,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.started.connect(self._on_worker_started)
        worker.succeeded.connect(self._on_worker_succeeded)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(thread.quit, type=QtCore.Qt.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._active_session = session
        self._thread = thread
        self._worker = worker
        self._request_ids.add(request.request_id)
        self.busy_changed.emit(True)
        try:
            thread.start()
        except Exception:
            self._active_session = None
            self._thread = None
            self._worker = None
            self.busy_changed.emit(False)
            raise
        return session

    @staticmethod
    def _validate_request(request):
        fixed = create_input_config(51_200)
        if (
            request.purpose != "calibration"
            or request.sample_rate != 51_200
            or request.target_samples != 512_000
            or len(request.channels) != 1
            or request.streaming
            or request.trim_samples != 0
            or request.device["input_config"].to_dict() != fixed
        ):
            raise ValueError("standalone calibration requires the fixed 51200 Hz profile")

    def _current(self, request_id=None):
        session = self._active_session
        if session is None:
            return None
        if request_id is not None and session.request.request_id != request_id:
            return None
        return session

    def _deliver(self, session, kind, payload=None):
        delivered = getattr(session, "_delivered", None)
        if delivered is None:
            delivered = session._delivered = set()
        if kind in delivered:
            return
        delivered.add(kind)
        callback = getattr(session.callbacks, kind)
        if callback is None:
            return
        try:
            if kind in ("started", "released"):
                callback(session)
            else:
                callback(session, payload)
        except Exception as error:
            self._report_callback_failure(session, kind, error)
            if kind == "result_ready":
                self._reject_result(session.request.request_id, str(error))

    def _report_callback_failure(self, session, kind, error):
        message = str(error)
        session.callback_diagnostics.append((kind, message))
        _LOGGER.exception("Calibration %s callback failed", kind)
        QtCore.QTimer.singleShot(
            0,
            lambda callback_kind=kind, diagnostic=message: (
                self._emit_callback_failure(callback_kind, diagnostic)
            ),
        )

    def _emit_callback_failure(self, kind, message):
        try:
            self.callback_failed.emit(kind, message)
        except Exception:
            # This is a one-way presentation boundary.  Never recursively emit
            # another diagnostic or alter a completed release transition.
            _LOGGER.exception("Calibration callback diagnostic listener failed")

    @QtCore.pyqtSlot()
    def _on_worker_started(self):
        session = self._current()
        if session is None:
            return
        session.state = "recording"
        self._deliver(session, "started")

    @QtCore.pyqtSlot(object)
    def _on_worker_succeeded(self, audio):
        session = self._current()
        if session is None or session._terminal_delivered:
            return
        if session.cancel_requested:
            self._terminal_cancel(session)
            return
        session.audio = audio
        session.descriptor = audio.descriptor
        session.state = "delivering"
        self._deliver(session, "result_ready", audio)

    @QtCore.pyqtSlot(object, bool)
    def _on_worker_failed(self, failure, cleanup_uncertain):
        session = self._current()
        if session is None or session._terminal_delivered:
            return
        session.failure = failure
        session._cleanup_uncertain = bool(cleanup_uncertain)
        if cleanup_uncertain:
            session.release_error = failure.message
        session.state = "failed"
        session._terminal_delivered = True
        self._deliver(session, "failed", failure)
        self.operation_failed.emit(str(failure.message))
        self._try_release(session)

    @QtCore.pyqtSlot(object, object)
    def _on_worker_cancelled(self, cancelled, cleanup_error):
        session = self._current()
        if session is None or session._terminal_delivered:
            return
        session._cleanup_uncertain = cleanup_error is not None
        if cleanup_error is not None:
            session.release_error = str(cleanup_error)
        session.state = "cancelled"
        session._terminal_delivered = True
        self._deliver(session, "cancelled", cancelled)
        self._try_release(session)

    @QtCore.pyqtSlot()
    def _on_thread_finished(self):
        session = self._current()
        if session is None:
            return
        session._thread_finished = True
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._try_release(session)

    def _cancel(self, request_id):
        self._require_gui_thread()
        session = self._current(request_id)
        if session is None or session._terminal_delivered or session.cancel_requested:
            return
        session.cancel_requested = True
        session._cancel_event.set()
        if session.state == "delivering":
            self._terminal_cancel(session)

    def _terminal_cancel(self, session):
        if session._terminal_delivered:
            return
        session.state = "cancelled"
        session._terminal_delivered = True
        self._deliver(
            session,
            "cancelled",
            RecordingCancelled(
                request_id=session.request.request_id,
                path=session.request.path,
                raw_frames=0,
                final_frames=0,
                handles_released=True,
            ),
        )
        self._try_release(session)

    def _accept_result(self, request_id):
        self._require_gui_thread()
        session = self._current(request_id)
        if (
            session is None
            or session.state != "delivering"
            or session.cancel_requested
            or session._decision_made
        ):
            return
        session._decision_made = True
        session.state = "completed"
        session._terminal_delivered = True
        self._deliver(session, "accepted", session.audio)
        self._try_release(session)

    def _reject_result(self, request_id, reason):
        self._require_gui_thread()
        session = self._current(request_id)
        if session is None or session.state != "delivering" or session._decision_made:
            return
        session._decision_made = True
        failure = RecordingFailure(
            request_id=session.request.request_id,
            stage="delivery",
            path=session.request.path,
            message=str(reason),
            handles_released=True,
        )
        session.failure = failure
        session.state = "failed"
        session._terminal_delivered = True
        self._deliver(session, "failed", failure)
        self._try_release(session)

    def _try_release(self, session):
        if (
            self._current() is not session
            or not session._thread_finished
            or not session._terminal_delivered
        ):
            return
        if session._cleanup_uncertain:
            self._deliver(session, "release_failed", session.release_error)
            self.lockout_requested.emit(str(session.release_error))
        else:
            session.released.set()
            self._deliver(session, "released")
        self._active_session = None
        self.busy_changed.emit(False)
        callbacks, self._shutdown_callbacks = self._shutdown_callbacks, []
        for callback in callbacks:
            QtCore.QTimer.singleShot(0, callback)

    def shutdown(self, callback=None):
        self._require_gui_thread()
        if callback is not None:
            if self.service.busy:
                self._shutdown_callbacks.append(callback)
            else:
                QtCore.QTimer.singleShot(0, callback)
        self._closed = True
        if not self._shutdown_emitted:
            self._shutdown_emitted = True
            self.shutting_down.emit()
        session = self._current()
        if session is not None:
            self._cancel(session.request.request_id)


class StandaloneVEInputCalibrationDialog(QtWidgets.QDialog):
    """Modal input-only shell exposing Calibration, Reset, and Close."""

    calibration_state_changed = QtCore.pyqtSignal(bool)
    calibration_busy_changed = QtCore.pyqtSignal(bool)
    calibration_failed = QtCore.pyqtSignal(str)
    hardware_lockout_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        device_snapshot,
        input_channels,
        recording_bridge,
        ve_profile_store,
        ve_calibration_store,
        parent=None,
        input_calibration_factory=None,
    ):
        super().__init__(parent)
        if input_calibration_factory is None:
            from ui.calibration_window import InputCalibration

            input_calibration_factory = InputCalibration
        self.recording_bridge = recording_bridge
        self._close_pending = False
        self._dialog_completed = False
        self._hardware_locked = False
        self._intended_result = None
        self.setModal(True)
        self.setWindowTitle("VE Input Calibration")
        self.input_calibration = input_calibration_factory(
            input_device=device_snapshot,
            input_channels=list(input_channels),
            recording_bridge=recording_bridge,
            ve_profile_store=ve_profile_store,
            ve_calibration_store=ve_calibration_store,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.input_calibration)
        buttons = QtWidgets.QHBoxLayout()
        self.calibration_button = QtWidgets.QPushButton("Calibration")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.close_button = QtWidgets.QPushButton("Close")
        buttons.addWidget(self.calibration_button)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.calibration_button.clicked.connect(
            self.input_calibration.clicked_calibration
        )
        self.reset_button.clicked.connect(self.input_calibration.reset_btn_clicked)
        self.close_button.clicked.connect(self.close)
        self.input_calibration.calibration_state_changed.connect(
            self.calibration_state_changed
        )
        self.input_calibration.calibration_availability_changed.connect(
            self._sync_buttons
        )
        recording_bridge.busy_changed.connect(self._on_busy_changed)
        recording_bridge.operation_failed.connect(self.calibration_failed)
        recording_bridge.lockout_requested.connect(self.hardware_lockout_requested)
        self._sync_buttons()

    @QtCore.pyqtSlot()
    def _sync_buttons(self):
        enabled = (
            not self.recording_bridge.service.busy
            and not self._close_pending
            and not self._hardware_locked
        )
        available = bool(
            getattr(self.input_calibration, "calibration_available", True)
        )
        self.calibration_button.setEnabled(enabled and available)
        self.reset_button.setEnabled(enabled and available)
        self.close_button.setEnabled(not self._hardware_locked)

    @QtCore.pyqtSlot(bool)
    def _on_busy_changed(self, busy):
        self.calibration_busy_changed.emit(bool(busy))
        self._sync_buttons()

    def lock_hardware_uncertainty(self, diagnostic):
        """Permanently seal and close this dialog after uncertain SDK cleanup."""
        if self._hardware_locked:
            return
        self._hardware_locked = True
        self._close_pending = True
        self._intended_result = int(QtWidgets.QDialog.Rejected)
        self.setWindowTitle(f"VE Input Calibration - Locked: {diagnostic}")
        self._sync_buttons()
        self.recording_bridge.shutdown(callback=self._complete_shutdown)

    def _complete_shutdown(self):
        if self._dialog_completed:
            return
        self.input_calibration.close_recording()
        self._dialog_completed = True
        QtWidgets.QDialog.done(self, self._intended_result)

    def _request_exit(self, result):
        if self._dialog_completed or self._close_pending:
            return
        self._close_pending = True
        self._intended_result = int(result)
        self._sync_buttons()
        self.recording_bridge.shutdown(callback=self._complete_shutdown)

    def done(self, result):
        self._request_exit(result)

    def accept(self):
        self._request_exit(QtWidgets.QDialog.Accepted)

    def reject(self):
        self._request_exit(QtWidgets.QDialog.Rejected)

    def closeEvent(self, event):
        if self._dialog_completed:
            event.accept()
            return
        self._request_exit(QtWidgets.QDialog.Rejected)
        event.ignore()


def create_input_only_calibration_dialog(
    *,
    device_snapshot,
    input_channels,
    ve_calibration_store,
    client_factory=None,
    parent=None,
):
    """Construct one dialog with all mandatory standalone dependencies."""
    bridge = InProcessCalibrationBridge(
        parent=parent,
        client_factory=client_factory,
    )
    return StandaloneVEInputCalibrationDialog(
        device_snapshot=device_snapshot,
        input_channels=input_channels,
        recording_bridge=bridge,
        ve_profile_store=Fixed51200VEProfileStore(),
        ve_calibration_store=ve_calibration_store,
        parent=parent,
    )
