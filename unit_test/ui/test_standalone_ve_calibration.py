import threading
import time
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QPushButton, QWidget

from base.recording_process_protocol import FrozenConfig, RecordingRequest, RecordingResult
from base.recording_result_reader import RecordingAudio
from base.recording_service import RecordingCallbacks
from base.ve3668n_input import create_input_config
from ui.recording_service_bridge import RecordingProcessorFacade
from ui.standalone_ve_calibration import (
    Fixed51200VEProfileStore,
    InProcessCalibrationBridge,
    StandaloneVEInputCalibrationDialog,
)


def _device():
    return {
        "backend": "vkinging",
        "model": "VE3668N",
        "machine_id": "MID-001",
        "name": "Dev1",
        "address": "192.0.2.1",
        "physical_channels": (0, 2),
        "max_input_channels": 3,
        "available": True,
        "input_config": create_input_config(48_000),
    }


def _request(channel=2):
    device = {**_device(), "input_config": create_input_config(51_200)}
    return RecordingRequest(
        request_id="calibration-1",
        purpose="calibration",
        sample_rate=51_200,
        target_samples=512_000,
        channels=(channel,),
        device=device,
        path=str(__file__),
        streaming=False,
        trim_samples=0,
        monitor={},
        calibration_metadata=None,
        validation_thresholds={},
    )


def _pump_until(app, predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            pytest.fail("Qt calibration operation timed out")
        time.sleep(0.001)
    app.processEvents()


def test_fixed_profile_observes_only_validated_fixed_profile_and_propagates():
    observed = []
    calibrations = SimpleNamespace(observe=lambda device: observed.append(device) or {})
    store = Fixed51200VEProfileStore()

    profile = store.load(_device(), calibrations)

    assert profile == create_input_config(51_200)
    assert observed == [{**_device(), "input_config": create_input_config(51_200)}]
    assert observed[0] is not _device()

    error = OSError("calibration store unreadable")
    failing = SimpleNamespace(observe=mock.Mock(side_effect=error))
    with pytest.raises(OSError, match="unreadable"):
        store.load(_device(), failing)
    failing.observe.assert_called_once()


class _AcquireClient:
    def __init__(
        self,
        *,
        outcome="success",
        samples=None,
        model="VE3668N",
        machine_id="MID-001",
        routes=("Rack-A/AIN1", "Rack-A/AIN3"),
    ):
        self.outcome = outcome
        self.samples = samples
        self.model = model
        self.machine_id = machine_id
        self.routes = routes
        self.calls = []
        self.thread_ids = []

    def __enter__(self):
        self.calls.append("enter")
        self.thread_ids.append(threading.get_ident())
        return self

    def __exit__(self, *_args):
        self.calls.append("exit")
        self.thread_ids.append(threading.get_ident())
        if self.outcome == "cleanup":
            error = OSError("VkDaqClearTask failed")
            error.ownership_uncertain = True
            raise error

    def select_device(self, selector):
        self.calls.append(("select", selector))
        self.thread_ids.append(threading.get_ident())
        return SimpleNamespace(index=0, name="Dev1", address="192.0.2.1")

    def list_devices(self):
        self.calls.append("list_devices")
        self.thread_ids.append(threading.get_ident())
        return (SimpleNamespace(index=0, name="Dev1", address="192.0.2.1"),)

    def get_device_attribute(self, selected, attribute):
        self.calls.append(("get_device_attribute", selected.name, attribute))
        self.thread_ids.append(threading.get_ident())
        return self.model if attribute == "Model" else self.machine_id

    def list_channels(self):
        self.calls.append("list_channels")
        self.thread_ids.append(threading.get_ident())
        return self.routes

    def acquire(self, **kwargs):
        self.calls.append(("acquire", kwargs))
        self.thread_ids.append(threading.get_ident())
        if self.outcome == "failure":
            raise OSError("capture failed")
        values = self.samples
        if values is None:
            values = np.linspace(-0.25, 0.25, 512_000, dtype=np.float32)
        return SimpleNamespace(
            channels=tuple(kwargs["channels"]),
            mode="iepe_voltage",
            sample_rate=51_200.0,
            requested_samples_per_channel=512_000,
            actual_samples_per_channel=len(values),
            samples=(values,),
        )


def _run_bridge(app, client, *, decision="accept"):
    events = []
    callback_threads = []
    bridge = InProcessCalibrationBridge(client_factory=lambda: client)
    bridge._test_terminal_busy = []
    bridge._test_lockouts = []
    bridge.lockout_requested.connect(bridge._test_lockouts.append)
    main_thread = threading.get_ident()

    def remember(name):
        def callback(session, payload=None):
            callback_threads.append(threading.get_ident())
            events.append((name, payload))
            if name in ("released", "release_failed"):
                bridge._test_terminal_busy.append(bridge.service.busy)
            if name == "result_ready":
                if decision == "accept":
                    session.accept_result()
                elif decision == "reject":
                    session.reject_result("bad signal")
        return callback

    callbacks = RecordingCallbacks(
        started=remember("started"),
        result_ready=remember("result_ready"),
        accepted=remember("accepted"),
        failed=remember("failed"),
        cancelled=remember("cancelled"),
        released=remember("released"),
        release_failed=remember("release_failed"),
    )
    request = _request()
    session = bridge.start(request, callbacks)
    assert bridge.service.busy
    assert session.request is request
    assert isinstance(session.request.device, FrozenConfig)
    facade = RecordingProcessorFacade(session)
    assert facade.target_samples == 512_000
    assert facade.sample_rate == 51_200
    terminal = {"released", "release_failed"}
    _pump_until(app, lambda: any(name in terminal for name, _ in events))
    assert callback_threads and set(callback_threads) == {main_thread}
    return bridge, session, facade, events


def test_calibration_bridge_success_exact_capture_audio_order_and_busy(ui_qapp):
    client = _AcquireClient()
    bridge, session, facade, events = _run_bridge(ui_qapp, client)

    assert [name for name, _ in events] == [
        "started", "result_ready", "accepted", "released"
    ]
    audio = events[1][1]
    assert isinstance(audio, RecordingAudio)
    assert isinstance(audio.descriptor, RecordingResult)
    assert audio.descriptor == RecordingResult(
        request_id="calibration-1",
        purpose="calibration",
        path=str(__file__),
        sample_rate=51_200,
        channels=(2,),
        raw_frames=512_000,
        final_frames=512_000,
        metadata_appended=False,
        handles_released=True,
    )
    assert audio.multi.shape == (512_000, 1)
    assert audio.mono.shape == (512_000,)
    assert audio.multi.dtype == audio.mono.dtype == np.float32
    assert np.isfinite(audio.multi).all()
    np.testing.assert_array_equal(audio.mono, audio.multi[:, 0])
    facade.set_recorded_audio(audio)
    np.testing.assert_array_equal(facade.get_recorded_data(), audio.mono)
    assert session.state == "completed"
    assert session.released.is_set()
    assert session.release_error is None
    assert not bridge.service.busy
    assert bridge._test_terminal_busy == [True]
    acquire = next(value for value in client.calls if isinstance(value, tuple) and value[0] == "acquire")
    assert acquire[1] == {
        "channels": ("Rack-A/AIN3",),
        "mode": "iepe_voltage",
        "sample_rate": 51_200,
        "samples_per_channel": 512_000,
        "min_value": -10.0,
        "max_value": 10.0,
        "terminal": "single_ended",
        "timeout": 12.0,
        "sensitivity": 1000.0,
    }
    assert client.thread_ids and set(client.thread_ids) == {client.thread_ids[0]}
    assert client.thread_ids[0] != threading.get_ident()


@pytest.mark.parametrize(
    ("client_kwargs", "diagnostic"),
    [
        ({"model": "VE3668N-X"}, "model"),
        ({"machine_id": "MID-CHANGED"}, "machine"),
        ({"routes": ("Rack-A/AIN1", "Rack-A/AIN2")}, "route"),
        ({"routes": ("Rack-A/AIN1",)}, "route"),
        ({"routes": ("Rack-A/AIN3", "Rack-A/AIN1")}, "order"),
        ({"routes": ("Rack-A/AIN1", "Rack-B/AIN1")}, "duplicate"),
        ({"routes": ("Rack-A/AIN01", "Rack-A/AIN3")}, "malformed"),
    ],
)
def test_calibration_bridge_revalidates_frozen_identity_and_routes_before_acquire(
    ui_qapp, client_kwargs, diagnostic
):
    client = _AcquireClient(**client_kwargs)
    _bridge, _session, _facade, events = _run_bridge(
        ui_qapp, client, decision="none"
    )

    assert [name for name, _ in events] == ["started", "failed", "released"]
    assert diagnostic in events[1][1].message.lower()
    assert not any(
        isinstance(call, tuple) and call[0] == "acquire" for call in client.calls
    )


def test_calibration_bridge_uses_exact_enumerated_authoritative_route(ui_qapp):
    client = _AcquireClient(routes=("routed-device/AIN1", "routed-device/AIN3"))

    _bridge, _session, _facade, _events = _run_bridge(ui_qapp, client)

    acquire = next(call for call in client.calls if isinstance(call, tuple) and call[0] == "acquire")
    assert acquire[1]["channels"] == ("routed-device/AIN3",)
    assert client.calls.index("list_devices") < client.calls.index(("select", 0))
    assert client.calls.index("list_channels") < client.calls.index(acquire)


@pytest.mark.parametrize(
    ("outcome", "decision", "expected", "state"),
    [
        ("success", "reject", ["started", "result_ready", "failed", "released"], "failed"),
        ("failure", "none", ["started", "failed", "released"], "failed"),
        ("cleanup", "none", ["started", "failed", "release_failed"], "failed"),
    ],
)
def test_calibration_bridge_reject_failure_cleanup_orderings_and_idempotency(
    ui_qapp, outcome, decision, expected, state
):
    bridge, session, _facade, events = _run_bridge(
        ui_qapp, _AcquireClient(outcome=outcome), decision=decision
    )
    assert [name for name, _ in events] == expected
    assert session.state == state
    assert not bridge.service.busy
    if outcome == "cleanup":
        assert session.release_error == "VkDaqClearTask failed"
        assert not session.released.is_set()
        assert bridge._test_lockouts == ["VkDaqClearTask failed"]
    else:
        assert session.released.is_set()
        assert bridge._test_lockouts == []
    session.accept_result()
    session.accept_result()
    session.reject_result("late")
    session.cancel()
    ui_qapp.processEvents()
    assert [name for name, _ in events] == expected


def test_calibration_bridge_cancel_and_gui_thread_only_start(ui_qapp):
    bridge = InProcessCalibrationBridge(client_factory=lambda: _AcquireClient())
    errors = []

    def wrong_thread():
        try:
            bridge.start(_request(), RecordingCallbacks())
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "GUI thread" in str(errors[0])

    events = []
    session = bridge.start(
        _request(),
        RecordingCallbacks(
            started=lambda current: (events.append("started"), current.cancel()),
            result_ready=lambda *_: events.append("result_ready"),
            cancelled=lambda *_: events.append("cancelled"),
            released=lambda *_: events.append("released"),
        ),
    )
    session.cancel()
    session.cancel()
    _pump_until(ui_qapp, lambda: "released" in events)
    assert events == ["started", "cancelled", "released"]
    assert session.state == "cancelled"
    assert session.cancel_requested


def test_input_only_dialog_injects_exact_dependencies_and_no_process(ui_qapp, monkeypatch):
    import base.recording_service as main_service
    import ui.calibration_window as calibration_window
    import ui.recording_service_bridge as main_bridge

    forbidden = mock.Mock(side_effect=AssertionError("main recording architecture touched"))
    monkeypatch.setattr(main_service, "RecordingService", forbidden)
    monkeypatch.setattr(main_bridge, "RecordingServiceBridge", forbidden)
    monkeypatch.setattr(calibration_window, "RecordingService", forbidden)
    monkeypatch.setattr(calibration_window, "RecordingServiceBridge", forbidden)
    monkeypatch.setattr(calibration_window, "OutputCalibration", forbidden)
    monkeypatch.setattr("multiprocessing.Process", forbidden)
    captured = {}

    class InputFake(QWidget):
        def __init__(self, **kwargs):
            super().__init__()
            self.calibration_state_changed = SimpleNamespace(connect=lambda callback: None)
            self.calibration_availability_changed = SimpleNamespace(connect=lambda callback: None)
            self.clicked_calibration = mock.Mock()
            self.reset_btn_clicked = mock.Mock()
            self.cancel_calibration = mock.Mock()
            self.close_recording = mock.Mock()
            captured.update(kwargs)

    bridge = InProcessCalibrationBridge(client_factory=lambda: _AcquireClient())
    profile = Fixed51200VEProfileStore()
    calibrations = object()
    dialog = StandaloneVEInputCalibrationDialog(
        device_snapshot={**_device(), "input_config": create_input_config(51_200)},
        input_channels=[0, 2],
        recording_bridge=bridge,
        ve_profile_store=profile,
        ve_calibration_store=calibrations,
        input_calibration_factory=InputFake,
    )

    assert captured == {
        "input_device": {**_device(), "input_config": create_input_config(51_200)},
        "input_channels": [0, 2],
        "recording_bridge": bridge,
        "ve_profile_store": profile,
        "ve_calibration_store": calibrations,
    }
    assert [button.text() for button in dialog.findChildren(QPushButton)] == [
        "Calibration", "Reset", "Close"
    ]
    forbidden.assert_not_called()
    dialog.close()


def test_input_only_close_seals_bridge_until_worker_release_and_finishes_once(ui_qapp):
    entered_acquire = threading.Event()
    release_acquire = threading.Event()

    class BlockingClient(_AcquireClient):
        def acquire(self, **kwargs):
            entered_acquire.set()
            if not release_acquire.wait(3):
                raise TimeoutError("test did not release calibration acquisition")
            return super().acquire(**kwargs)

    class InputFake(QWidget):
        def __init__(self, **_kwargs):
            super().__init__()
            self.calibration_state_changed = SimpleNamespace(connect=lambda callback: None)
            self.calibration_availability_changed = SimpleNamespace(connect=lambda callback: None)
            self.clicked_calibration = mock.Mock()
            self.reset_btn_clicked = mock.Mock()
            self.cancel_calibration = mock.Mock()
            self.close_recording = mock.Mock()

    close_observations = []

    class TrackingDialog(StandaloneVEInputCalibrationDialog):
        def closeEvent(self, event):
            super().closeEvent(event)
            session = self.recording_bridge._active_session
            close_observations.append(
                (
                    event.isAccepted(),
                    self.recording_bridge.service.busy,
                    None if session is None else session._thread_finished,
                )
            )

    client = BlockingClient()
    bridge = InProcessCalibrationBridge(client_factory=lambda: client)
    shutdown_signals = []
    bridge.shutting_down.connect(lambda: shutdown_signals.append(True))
    dialog = TrackingDialog(
        device_snapshot={**_device(), "input_config": create_input_config(51_200)},
        input_channels=[0, 2],
        recording_bridge=bridge,
        ve_profile_store=Fixed51200VEProfileStore(),
        ve_calibration_store=object(),
        input_calibration_factory=InputFake,
    )
    session = bridge.start(_request(), RecordingCallbacks())
    _pump_until(ui_qapp, entered_acquire.is_set)
    dialog.show()
    finished = []
    dialog.finished.connect(finished.append)

    assert dialog.close() is False
    second = QCloseEvent()
    dialog.closeEvent(second)
    assert not second.isAccepted()
    assert shutdown_signals == [True]
    assert session.cancel_requested
    with pytest.raises(RuntimeError, match="shutting down"):
        bridge.start(_request(), RecordingCallbacks())
    assert all(not accepted for accepted, _busy, _finished in close_observations)
    assert finished == []

    release_acquire.set()
    _pump_until(ui_qapp, lambda: bool(finished))

    assert shutdown_signals == [True]
    assert finished == [QDialog.Rejected]
    assert dialog.result() == QDialog.Rejected
    assert all(not accepted for accepted, _busy, _finished in close_observations)
    assert not dialog.isVisible()
    assert session._thread_finished
    dialog.input_calibration.close_recording.assert_called_once()


def test_input_only_hardware_lockout_stays_disabled_after_busy_false_and_closes(
    ui_qapp
):
    class InputFake(QWidget):
        def __init__(self, **_kwargs):
            super().__init__()
            self.calibration_state_changed = SimpleNamespace(connect=lambda callback: None)
            self.calibration_availability_changed = SimpleNamespace(connect=lambda callback: None)
            self.calibration_available = True
            self.clicked_calibration = mock.Mock()
            self.reset_btn_clicked = mock.Mock()
            self.close_recording = mock.Mock()

    bridge = InProcessCalibrationBridge(client_factory=lambda: _AcquireClient())
    dialog = StandaloneVEInputCalibrationDialog(
        device_snapshot={**_device(), "input_config": create_input_config(51_200)},
        input_channels=[0, 2],
        recording_bridge=bridge,
        ve_profile_store=Fixed51200VEProfileStore(),
        ve_calibration_store=object(),
        input_calibration_factory=InputFake,
    )
    dialog.show()
    finished = []
    dialog.finished.connect(finished.append)

    dialog.lock_hardware_uncertainty("ClearTask ownership uncertain")
    bridge.busy_changed.emit(False)
    ui_qapp.processEvents()

    assert not dialog.calibration_button.isEnabled()
    assert not dialog.reset_button.isEnabled()
    assert not dialog.close_button.isEnabled()
    with pytest.raises(RuntimeError, match="shutting down"):
        bridge.start(_request(), RecordingCallbacks())
    _pump_until(ui_qapp, lambda: bool(finished))
    assert finished == [QDialog.Rejected]
    assert not dialog.isVisible()


@pytest.mark.parametrize(
    ("first_exit", "expected_result"),
    [
        ("reject", QDialog.Rejected),
        ("accept", QDialog.Accepted),
        ("done", 37),
    ],
)
def test_input_only_reject_or_done_waits_for_shutdown_and_preserves_first_result(
    ui_qapp, first_exit, expected_result
):
    entered_acquire = threading.Event()
    release_acquire = threading.Event()

    class BlockingClient(_AcquireClient):
        def acquire(self, **kwargs):
            entered_acquire.set()
            if not release_acquire.wait(3):
                raise TimeoutError("test did not release calibration acquisition")
            return super().acquire(**kwargs)

    class InputFake(QWidget):
        def __init__(self, **_kwargs):
            super().__init__()
            self.calibration_state_changed = SimpleNamespace(connect=lambda callback: None)
            self.calibration_availability_changed = SimpleNamespace(connect=lambda callback: None)
            self.clicked_calibration = mock.Mock()
            self.reset_btn_clicked = mock.Mock()
            self.cancel_calibration = mock.Mock()
            self.close_recording = mock.Mock()

    bridge = InProcessCalibrationBridge(client_factory=lambda: BlockingClient())
    shutdown_signals = []
    bridge.shutting_down.connect(lambda: shutdown_signals.append(True))
    dialog = StandaloneVEInputCalibrationDialog(
        device_snapshot={**_device(), "input_config": create_input_config(51_200)},
        input_channels=[0, 2],
        recording_bridge=bridge,
        ve_profile_store=Fixed51200VEProfileStore(),
        ve_calibration_store=object(),
        input_calibration_factory=InputFake,
    )
    finished = []
    dialog.finished.connect(finished.append)
    session = bridge.start(_request(), RecordingCallbacks())
    _pump_until(ui_qapp, entered_acquire.is_set)
    dialog.show()

    if first_exit == "reject":
        QTest.keyClick(dialog, Qt.Key_Escape)
    elif first_exit == "accept":
        dialog.accept()
    else:
        dialog.done(37)
    dialog.done(99)
    dialog.reject()
    dialog.close()
    ui_qapp.processEvents()

    assert dialog.isVisible()
    assert finished == []
    assert shutdown_signals == [True]
    assert session.cancel_requested
    with pytest.raises(RuntimeError, match="shutting down"):
        bridge.start(_request(), RecordingCallbacks())

    release_acquire.set()
    _pump_until(ui_qapp, lambda: bool(finished))

    assert finished == [expected_result]
    assert dialog.result() == expected_result
    assert not dialog.isVisible()
    assert shutdown_signals == [True]
    assert session._thread_finished
    dialog.input_calibration.close_recording.assert_called_once()


@pytest.mark.parametrize(
    ("failing_kind", "outcome", "cancel", "expected_order"),
    [
        ("started", "success", False, ["started", "result_ready", "accepted", "released"]),
        ("accepted", "success", False, ["started", "result_ready", "accepted", "released"]),
        ("failed", "failure", False, ["started", "failed", "released"]),
        ("cancelled", "success", True, ["started", "cancelled", "released"]),
        ("released", "success", False, ["started", "result_ready", "accepted", "released"]),
        ("release_failed", "cleanup", False, ["started", "failed", "release_failed"]),
    ],
)
def test_calibration_bridge_callback_failure_is_diagnostic_and_keeps_release_semantics(
    ui_qapp, caplog, failing_kind, outcome, cancel, expected_order
):
    bridge = InProcessCalibrationBridge(
        client_factory=lambda: _AcquireClient(outcome=outcome)
    )
    diagnostics = []
    lockouts = []
    bridge.callback_failed.connect(
        lambda kind, message: diagnostics.append((kind, message))
    )
    bridge.lockout_requested.connect(lockouts.append)
    attempts = []

    def callback(kind):
        def invoke(session, _payload=None):
            attempts.append(kind)
            if kind == "result_ready":
                session.accept_result()
            if kind == failing_kind:
                raise RuntimeError(f"boom-{kind}")
        return invoke

    callbacks = RecordingCallbacks(
        started=callback("started"),
        result_ready=callback("result_ready"),
        accepted=callback("accepted"),
        failed=callback("failed"),
        cancelled=callback("cancelled"),
        released=callback("released"),
        release_failed=callback("release_failed"),
    )
    session = bridge.start(_request(), callbacks)
    if cancel:
        session.cancel()
    _pump_until(ui_qapp, lambda: not bridge.service.busy)
    _pump_until(ui_qapp, lambda: bool(diagnostics))

    assert attempts == expected_order
    assert diagnostics == [(failing_kind, f"boom-{failing_kind}")]
    assert session.callback_diagnostics == diagnostics
    assert f"{failing_kind} callback failed" in caplog.text
    assert attempts.count(failing_kind) == 1
    assert not bridge.service.busy
    if outcome == "cleanup":
        assert session.release_error == "VkDaqClearTask failed"
        assert lockouts == ["VkDaqClearTask failed"]
        assert not session.released.is_set()
    else:
        assert lockouts == []
        assert session.released.is_set()


def test_calibration_bridge_result_ready_exception_is_diagnostic_then_rejected(ui_qapp):
    bridge = InProcessCalibrationBridge(client_factory=lambda: _AcquireClient())
    attempts = []
    diagnostics = []
    bridge.callback_failed.connect(
        lambda kind, message: diagnostics.append((kind, message))
    )

    def result_ready(_session, _audio):
        attempts.append("result_ready")
        raise RuntimeError("boom-result_ready")

    session = bridge.start(
        _request(),
        RecordingCallbacks(
            started=lambda _session: attempts.append("started"),
            result_ready=result_ready,
            failed=lambda _session, _failure: attempts.append("failed"),
            released=lambda _session: attempts.append("released"),
        ),
    )
    _pump_until(ui_qapp, lambda: not bridge.service.busy)
    _pump_until(ui_qapp, lambda: bool(diagnostics))

    assert attempts == ["started", "result_ready", "failed", "released"]
    assert diagnostics == [("result_ready", "boom-result_ready")]
    assert session.failure.message == "boom-result_ready"
    assert session.state == "failed"
    assert session.released.is_set()


def test_calibration_bridge_diagnostic_listener_failure_cannot_break_release(
    ui_qapp, monkeypatch
):
    bridge = InProcessCalibrationBridge(client_factory=lambda: _AcquireClient())
    listener_errors = []
    monkeypatch.setattr(
        sys,
        "excepthook",
        lambda error_type, error, traceback: listener_errors.append(
            (error_type, str(error), traceback)
        ),
    )

    def broken_listener(_kind, _message):
        raise RuntimeError("diagnostic listener failed")

    bridge.callback_failed.connect(broken_listener)
    session = bridge.start(
        _request(),
        RecordingCallbacks(
            result_ready=lambda current, _audio: current.accept_result(),
            released=lambda _session: (_ for _ in ()).throw(
                RuntimeError("boom-released")
            ),
        ),
    )
    _pump_until(ui_qapp, lambda: not bridge.service.busy)
    _pump_until(ui_qapp, lambda: bool(listener_errors))

    assert session.released.is_set()
    assert not bridge.service.busy
    assert session.callback_diagnostics == [("released", "boom-released")]
    assert listener_errors[0][0] is RuntimeError
    assert listener_errors[0][1] == "diagnostic listener failed"


def test_calibration_bridge_cancel_with_uncertain_cleanup_reports_unreleased_cancel(
    ui_qapp,
):
    entered_acquire = threading.Event()
    release_acquire = threading.Event()

    class BlockingCleanupClient(_AcquireClient):
        def __init__(self):
            super().__init__(outcome="cleanup")

        def acquire(self, **kwargs):
            entered_acquire.set()
            if not release_acquire.wait(3):
                raise TimeoutError("test did not release calibration acquisition")
            return super().acquire(**kwargs)

    bridge = InProcessCalibrationBridge(client_factory=BlockingCleanupClient)
    events = []
    lockouts = []
    bridge.lockout_requested.connect(lockouts.append)
    session = bridge.start(
        _request(),
        RecordingCallbacks(
            started=lambda _session: events.append(("started", None)),
            cancelled=lambda _session, payload: events.append(("cancelled", payload)),
            release_failed=lambda _session, error: events.append(("release_failed", error)),
        ),
    )
    _pump_until(ui_qapp, entered_acquire.is_set)
    session.cancel()
    release_acquire.set()
    _pump_until(ui_qapp, lambda: not bridge.service.busy)

    assert [kind for kind, _payload in events] == [
        "started", "cancelled", "release_failed"
    ]
    cancelled = events[1][1]
    assert cancelled.handles_released is False
    assert events[2][1] == "VkDaqClearTask failed"
    assert session.release_error == "VkDaqClearTask failed"
    assert not session.released.is_set()
    assert lockouts == ["VkDaqClearTask failed"]
    assert not bridge.service.busy
