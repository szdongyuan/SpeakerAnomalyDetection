import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile
import soundfile as sf

import record_vkinging_continuous as recorder
from record_vkinging_continuous import SegmentAccumulator, WavPublisher


def _button_snapshot(window):
    return {
        "start": window.start_button.isEnabled(),
        "stop": window.stop_button.isEnabled(),
        "calibration": window.calibration_button.isEnabled(),
        "parameters": window.parameters_button.isEnabled(),
        "analyze": window.analyze_button.isEnabled(),
        "retry_visible": not window.retry_button.isHidden(),
        "retry_enabled": window.retry_button.isEnabled(),
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("DISCOVERING", (False, False, False, True, False, False, False)),
        ("UNAVAILABLE", (False, False, False, True, False, True, True)),
        ("IDLE", (True, False, True, True, False, False, False)),
        ("RECORDING", (False, True, False, False, False, False, False)),
        ("FINALIZING", (False, False, False, False, False, False, False)),
        ("READY", (True, False, True, True, True, False, False)),
        ("CALIBRATING", (False, False, False, False, False, False, False)),
        ("ANALYZING", (False, False, False, False, False, False, False)),
        ("ERROR", (False, False, False, False, False, False, False)),
        ("LOCKED", (False, False, False, False, False, False, False)),
        ("CLOSING", (False, False, False, False, False, False, False)),
    ],
)
def test_window_state_button_policy_covers_every_nominal_state(state, expected):
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    try:
        window._device_snapshot = {"physical_channels": (0, 2)}
        window._channel_routes = ("Dev1/AIN1", "Dev1/AIN3")
        window._current_run_path = Path("current.wav")
        window._spl_controller = SimpleNamespace(analyze_available=True)
        window._transition(recorder.WindowState[state])
        app.processEvents()

        actual = _button_snapshot(window)
        assert tuple(actual.values()) == expected
        if state == "LOCKED":
            assert "exit" in window.status_label.text().lower()
    finally:
        window._allow_close = True
        window.close()


def test_spl_parameters_click_invokes_controller_in_every_enabled_discovery_state():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class SplController:
        analyze_available = False

        def __init__(self):
            self.open_calls = 0

        def open_parameters(self):
            self.open_calls += 1

    spl = SplController()
    window = visual.WaveformWindow(
        auto_discover=False,
        spl_controller=spl,
    )
    try:
        assert window.state is recorder.WindowState.DISCOVERING
        assert window.parameters_button.isEnabled()
        window.parameters_button.click()

        window._transition(recorder.WindowState.UNAVAILABLE)
        assert window.parameters_button.isEnabled()
        window.parameters_button.click()

        assert spl.open_calls == 2

        window._transition(recorder.WindowState.RECORDING)
        assert not window.parameters_button.isEnabled()
        window.parameters_button.click()
        assert spl.open_calls == 2
    finally:
        window._allow_close = True
        window.close()
        app.processEvents()


def test_window_state_startup_discovers_without_auto_record_and_unavailable_retry():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Discovery:
        def __init__(self):
            self.starts = 0
            self.retries = 0

        def start_discovery(self):
            self.starts += 1

        def retry_discovery(self):
            self.retries += 1

    discovery = Discovery()
    recording_calls = []
    window = visual.WaveformWindow(
        discovery_coordinator=discovery,
        recording_session_factory=lambda **kwargs: recording_calls.append(kwargs),
    )
    try:
        assert discovery.starts == 1
        assert recording_calls == []
        assert window.state is recorder.WindowState.DISCOVERING

        window.handle_discovery_failure("SDK offline")
        assert window.state is recorder.WindowState.UNAVAILABLE
        assert not window.retry_button.isHidden()
        window.retry_button.click()
        assert discovery.retries == 1
        assert window.state is recorder.WindowState.DISCOVERING
    finally:
        window._allow_close = True
        window.close()
        app.processEvents()


def test_window_state_recording_result_waits_for_thread_and_clears_old_target():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    old = Path("old.wav")
    window._device_snapshot = {"physical_channels": (0, 2)}
    window._channel_routes = ("Dev1/AIN1", "Dev1/AIN3")
    window._current_run_path = old
    window._transition(recorder.WindowState.READY)
    window._begin_recording_state()

    assert window._current_run_path is None
    assert window.state is recorder.WindowState.RECORDING
    assert window.countdown_label.text() == "05:00"

    result = recorder.VisualRecordingResult(
        path=Path("new.wav"), accepted_frames=102400, written_frames=102400
    )
    window.handle_recording_result(result)
    assert window.state is recorder.WindowState.FINALIZING
    assert window._current_run_path is None
    window.handle_recording_thread_finished()
    assert window.state is recorder.WindowState.READY
    assert window._current_run_path == Path("new.wav")

    window._begin_recording_state()
    window.handle_recording_result(
        recorder.VisualRecordingResult(
            path=None, accepted_frames=0, written_frames=0, no_audio=True
        )
    )
    window.handle_recording_thread_finished()
    assert window.state is recorder.WindowState.IDLE
    assert window._current_run_path is None

    window._allow_close = True
    window.close()
    app.processEvents()


def test_window_state_countdown_uses_accepted_frames_and_early_stop_duration():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    try:
        window._begin_recording_state()
        window.handle_accepted_frames(51_200 * 65)
        assert window.countdown_label.text() == "03:55"
        window.handle_accepted_frames(51_200 * 300)
        assert window.countdown_label.text() == "00:00"

        window._begin_recording_state()
        window.handle_accepted_frames(51_200 * 12)
        window.request_stop()
        assert window.state is recorder.WindowState.FINALIZING
        assert "00:12" in window.status_label.text()
        assert window.countdown_label.text() == "04:48"
    finally:
        window._allow_close = True
        window.close()
        app.processEvents()


def test_close_intent_during_recording_requests_one_stop_and_waits_for_finished():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Controller:
        def __init__(self):
            self.calls = 0

        def request_stop(self):
            self.calls += 1

    controller = Controller()
    window = visual.WaveformWindow(auto_discover=False)
    window.show()
    window._recording_controller = controller
    window._transition(recorder.WindowState.RECORDING)
    window.close()
    window.close()
    app.processEvents()
    assert controller.calls == 1
    assert window.state is recorder.WindowState.CLOSING
    assert window.isVisible()

    window.handle_recording_result(
        recorder.VisualRecordingResult(
            path=None, accepted_frames=0, written_frames=0, no_audio=True
        )
    )
    window.handle_recording_thread_finished()
    app.processEvents()
    assert window.state is recorder.WindowState.CLOSED
    assert not window.isVisible()


def test_close_intent_calibration_and_analysis_wait_for_async_terminal_boundary():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Bridge:
        def __init__(self):
            self.callbacks = []

        def shutdown(self, callback=None):
            self.callbacks.append(callback)

    bridge = Bridge()
    calibration = SimpleNamespace(recording_bridge=bridge, close=lambda: None)
    window = visual.WaveformWindow(auto_discover=False)
    window.show()
    window._calibration_dialog = calibration
    window._transition(recorder.WindowState.CALIBRATING)
    window.close()
    app.processEvents()
    assert window.state is recorder.WindowState.CLOSING
    assert len(bridge.callbacks) == 1
    assert window.isVisible()
    bridge.callbacks[0]()
    app.processEvents()
    assert window.state is recorder.WindowState.CLOSED

    class Analysis:
        def __init__(self):
            self.cancel_calls = 0
            self.analyzing = True

        def request_cancel(self):
            self.cancel_calls += 1

    analysis = Analysis()
    window = visual.WaveformWindow(auto_discover=False, spl_controller=analysis)
    window.show()
    window._transition(recorder.WindowState.ANALYZING)
    window.close()
    window.close()
    app.processEvents()
    assert analysis.cancel_calls == 1
    assert window.isVisible()
    analysis.analyzing = False
    window.handle_analysis_finished(SimpleNamespace(cancelled=True, failures=()))
    app.processEvents()
    assert window.state is recorder.WindowState.CLOSED


def test_window_state_calibration_lockout_retains_dialog_until_finished():
    from PyQt5 import QtCore

    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Dialog(QtCore.QObject):
        finished = QtCore.pyqtSignal(int)
        hardware_lockout_requested = QtCore.pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.lock_calls = []

        def lock_hardware_uncertainty(self, diagnostic):
            self.lock_calls.append(diagnostic)

    dialog = Dialog()
    window = visual.WaveformWindow(auto_discover=False)
    window._calibration_dialog = dialog
    dialog.hardware_lockout_requested.connect(window.handle_calibration_lockout)
    dialog.finished.connect(window.handle_calibration_finished)
    window._transition(recorder.WindowState.CALIBRATING)

    dialog.hardware_lockout_requested.emit("ClearTask ownership uncertain")
    app.processEvents()

    assert window.state is recorder.WindowState.LOCKED
    assert window._calibration_dialog is dialog
    assert dialog.lock_calls == ["ClearTask ownership uncertain"]

    dialog.finished.emit(0)
    app.processEvents()
    assert window._calibration_dialog is None
    assert window.state is recorder.WindowState.LOCKED
    window._allow_close = True
    window.close()


def test_window_state_cleanup_uncertainty_enters_locked_close_only():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    window._begin_recording_state()
    window.handle_recording_result(
        recorder.VisualRecordingResult(
            path=None,
            accepted_frames=1,
            written_frames=1,
            ownership_uncertain=True,
            failure=recorder.VisualRecordingFailure(
                "ClearTask failed", ownership_uncertain=True
            ),
        )
    )
    window.handle_recording_thread_finished()
    assert window.state is recorder.WindowState.LOCKED
    assert all(
        not button.isEnabled()
        for button in (
            window.start_button,
            window.stop_button,
            window.calibration_button,
            window.parameters_button,
            window.analyze_button,
        )
    )
    assert "ClearTask failed" in window.status_label.text()
    window._allow_close = True
    window.close()
    app.processEvents()


def test_window_state_ordinary_recording_failure_emits_error_then_recovers_idle():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    history = []
    window.state_changed.connect(history.append)
    window._begin_recording_state()
    window.handle_recording_result(
        recorder.VisualRecordingResult(
            path=None,
            accepted_frames=0,
            written_frames=0,
            failure=recorder.VisualRecordingFailure("capture failed"),
        )
    )
    window.handle_recording_thread_finished()

    assert history[-3:] == [
        recorder.WindowState.FINALIZING,
        recorder.WindowState.ERROR,
        recorder.WindowState.IDLE,
    ]
    assert "capture failed" in window.status_label.text()
    window._allow_close = True
    window.close()
    app.processEvents()


@pytest.mark.parametrize("starting_state", ["IDLE", "READY", "LOCKED"])
def test_close_intent_every_terminal_state_emits_closing_then_closed(starting_state):
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    window = visual.WaveformWindow(auto_discover=False)
    history = []
    window.state_changed.connect(history.append)
    window._transition(recorder.WindowState[starting_state])
    history.clear()
    window.show()

    window.close()
    app.processEvents()

    assert history == [recorder.WindowState.CLOSING, recorder.WindowState.CLOSED]
    assert not window.isVisible()


def test_window_state_repeat_start_freezes_metadata_and_uses_fresh_exact_target_session():
    from PyQt5 import QtCore

    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Discovery:
        calibration_store = object()
        device_snapshot = {"physical_channels": (0, 2)}
        channel_routes = ("Dev1/AIN1", "Dev1/AIN3")
        selected_device = SimpleNamespace(index=0)

        def __init__(self):
            self.freezes = []

        def freeze_metadata_for_start(self):
            frozen = {"generation": len(self.freezes) + 1}
            self.freezes.append(frozen)
            return frozen

    class Spl:
        analyze_available = False

        def __init__(self):
            self.targets = []

        def set_current_run(self, path, channels):
            self.targets.append((path, tuple(channels)))
            self.analyze_available = path is not None

    class Worker(QtCore.QObject):
        accepted_frames = QtCore.pyqtSignal(int)
        result_ready = QtCore.pyqtSignal(object)
        finished = QtCore.pyqtSignal()

        def __init__(self, result):
            super().__init__()
            self.result = result

        @QtCore.pyqtSlot()
        def run(self):
            self.result_ready.emit(self.result)
            self.finished.emit()

    sessions = []

    def session_factory(**kwargs):
        index = len(sessions) + 1
        thread = visual.QThread()
        worker = Worker(
            recorder.VisualRecordingResult(
                path=Path(f"run-{index}.wav"),
                accepted_frames=index,
                written_frames=index,
            )
        )
        controller = recorder.StopController()
        sessions.append((thread, worker, controller, kwargs))
        return thread, worker, controller

    discovery = Discovery()
    spl = Spl()
    window = visual.WaveformWindow(
        discovery_coordinator=discovery,
        spl_controller=spl,
        recording_session_factory=session_factory,
        auto_discover=False,
    )
    window._device_snapshot = discovery.device_snapshot
    window._channel_routes = discovery.channel_routes
    window._selected_device = discovery.selected_device
    window._transition(recorder.WindowState.IDLE)

    try:
        window.start_recording()
        _pump_qt_until(app, lambda: window.state is recorder.WindowState.READY)
        window.start_recording()
        _pump_qt_until(
            app,
            lambda: window.state is recorder.WindowState.READY and len(sessions) == 2,
        )

        assert len({id(item[0]) for item in sessions}) == 2
        assert len({id(item[1]) for item in sessions}) == 2
        assert [item[3]["metadata"] for item in sessions] == discovery.freezes
        assert all(
            item[3]["target_frames"] == 51_200 * 300 for item in sessions
        )
        assert spl.targets == [
            (None, ()),
            (Path("run-1.wav"), (0, 2)),
            (None, ()),
            (Path("run-2.wav"), (0, 2)),
        ]
    finally:
        window._allow_close = True
        window.close()
        app.processEvents()


def test_window_state_thread_start_failure_never_constructs_publisher_and_recovers_error(
    monkeypatch, tmp_path
):
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])

    class Discovery:
        calibration_store = object()
        device_snapshot = {"physical_channels": (0,)}
        channel_routes = ("Dev1/AIN1",)
        selected_device = SimpleNamespace(index=0)

        @staticmethod
        def freeze_metadata_for_start():
            return {"frozen": True}

    class Spl:
        analyze_available = False

        @staticmethod
        def set_current_run(_path, _channels):
            return None

    publisher_calls = []
    monkeypatch.setattr(
        visual.QThread,
        "start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("thread refused start")),
    )
    window = visual.WaveformWindow(
        discovery_coordinator=Discovery(),
        spl_controller=Spl(),
        publisher_factory=lambda **kwargs: publisher_calls.append(kwargs),
        auto_discover=False,
    )
    window._device_snapshot = Discovery.device_snapshot
    window._channel_routes = Discovery.channel_routes
    window._selected_device = Discovery.selected_device
    window._transition(recorder.WindowState.IDLE)
    history = []
    window.state_changed.connect(history.append)

    window.start_recording()

    assert publisher_calls == []
    assert history[-2:] == [recorder.WindowState.ERROR, recorder.WindowState.IDLE]
    assert "thread refused start" in window.status_label.text()
    assert list(tmp_path.iterdir()) == []
    window._allow_close = True
    window.close()
    app.processEvents()


def test_recorder_worker_constructs_and_discards_publisher_on_worker_thread():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    main_thread = threading.get_ident()
    controller = recorder.StopController()
    controller.request_stop()
    mailbox = recorder.WaveformMailbox(chart_limit=1, capacity_samples=8)
    mailbox.configure(("Dev1/AIN1",))
    client = _VisualClient((), channels=("Dev1/AIN1",))
    publisher = _VisualPublisher()
    factory_threads = []

    def publisher_factory(**_kwargs):
        factory_threads.append(threading.get_ident())
        return publisher

    worker = visual.RecorderWorker(
        controller=controller,
        mailbox=mailbox,
        client_factory=lambda: client,
        device_selector=0,
        channel_routes=("Dev1/AIN1",),
        device_snapshot=None,
        metadata={"frozen": True},
        publisher_factory=publisher_factory,
        publisher_config={"output_directory": Path("unused")},
        target_frames=51_200 * 300,
    )
    thread = visual.QThread()
    finished = []
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, type=visual.Qt.DirectConnection)
    thread.finished.connect(lambda: finished.append(True))
    thread.start()
    _pump_qt_until(app, lambda: bool(finished))

    assert factory_threads and factory_threads[0] != main_thread
    assert publisher.discard_calls == 1
    assert worker.result.no_audio


def test_spl_config_controller_factory_wires_isolated_owner(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("ui.standalone_ve_spl.StandaloneSplController", factory)
    config_path = tmp_path / "standalone-spl.json"

    result = recorder.create_standalone_spl_controller(
        config_path=config_path,
        parent=None,
    )

    assert result is sentinel
    assert captured == {"config_path": config_path, "parent": None}


def test_standalone_calibration_factory_is_input_only_and_no_process(monkeypatch):
    captured = {}

    class DialogFake:
        pass

    monkeypatch.setattr(
        "ui.standalone_ve_calibration.create_input_only_calibration_dialog",
        lambda **kwargs: captured.update(kwargs) or DialogFake(),
    )
    client_factory = object()
    calibration_store = object()
    snapshot = {
        "backend": "vkinging",
        "physical_channels": (0, 2),
    }

    dialog = recorder.create_standalone_calibration_dialog(
        device_snapshot=snapshot,
        calibration_store=calibration_store,
        client_factory=client_factory,
    )

    assert isinstance(dialog, DialogFake)
    assert captured["device_snapshot"] is snapshot
    assert captured["input_channels"] == [0, 2]
    assert captured["ve_calibration_store"] is calibration_store
    assert captured["client_factory"] is client_factory


def test_recorder_constants_match_standalone_visual_contract():
    assert recorder.SAMPLE_RATE == 51200
    assert recorder.SEGMENT_SECONDS == 300
    assert recorder.VISUAL_CHUNK_FRAMES == 5120
    assert recorder.MODE == "iepe_voltage"
    assert recorder.IEPE_SENSITIVITY == 1000.0


def _direct_device(*, name="Dev1", address="192.0.2.1"):
    return SimpleNamespace(index=0, name=name, address=address)


def test_device_snapshot_maps_ordered_ain_routes_to_physical_channels():
    snapshot = recorder.resolve_ve_device_snapshot(
        _direct_device(),
        model=" VE3668N ",
        machine_id=" MID-001 ",
        routes=("Dev1/AIN1", "Dev1/DIO1", "Dev1/AIN3"),
    )

    assert snapshot == {
        "backend": "vkinging",
        "model": "VE3668N",
        "machine_id": "MID-001",
        "name": "Dev1",
        "address": "192.0.2.1",
        "physical_channels": (0, 2),
        "max_input_channels": 3,
        "available": True,
        "input_config": {
            "sample_rate": 51200,
            "input_mode": "IEPE",
            "unit": "V",
            "range_min": -10.0,
            "range_max": 10.0,
        },
    }


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        (("Dev1/AIN0",), "malformed"),
        (("Dev1/AIN01",), "malformed"),
        (("Dev1/AINx",), "malformed"),
        (("Dev1/AIN1/extra",), "malformed"),
        (("Dev1/AIN1", "Dev1/AIN1"), "duplicate"),
        (("Dev1/AIN1", "Dev1/AIN9"), "out of range"),
    ],
)
def test_device_snapshot_rejects_invalid_ain_routes(routes, message):
    with pytest.raises(ValueError, match=message):
        recorder.resolve_ve_device_snapshot(
            _direct_device(),
            model="VE3668N",
            machine_id="MID-001",
            routes=routes,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": ""}, "model"),
        ({"model": "VE3668N-extra"}, "model"),
        ({"machine_id": ""}, "machine_id"),
        ({"device": _direct_device(name="")}, "name"),
    ],
)
def test_device_snapshot_rejects_missing_or_unstable_identity(overrides, message):
    arguments = {
        "device": _direct_device(),
        "model": "VE3668N",
        "machine_id": "MID-001",
        "routes": ("Dev1/AIN1",),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        recorder.resolve_ve_device_snapshot(**arguments)


def test_device_snapshot_rejects_zero_usable_ain_channels():
    with pytest.raises(ValueError, match="analog input"):
        recorder.resolve_ve_device_snapshot(
            _direct_device(),
            model="VE3668N",
            machine_id="MID-001",
            routes=("Dev1/DIO1", "Dev1/TACH1"),
        )


def _pump_qt_until(app, predicate, *, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            pytest.fail("Qt operation did not finish before the test timeout")
        time.sleep(0.001)
    app.processEvents()


class _DiscoveryClient:
    def __init__(self, *, failure=None, machine_id="MID-001"):
        self.failure = failure
        self.machine_id = machine_id
        self.calls = []
        self.worker_thread_ids = []

    def __enter__(self):
        self.calls.append("enter")
        self.worker_thread_ids.append(threading.get_ident())
        return self

    def __exit__(self, *_args):
        self.calls.append("exit")
        self.worker_thread_ids.append(threading.get_ident())

    def _called(self, name):
        self.calls.append(name)
        self.worker_thread_ids.append(threading.get_ident())
        if self.failure is not None:
            error = self.failure
            self.failure = None
            raise error

    def list_devices(self):
        self._called("list_devices")
        return (_direct_device(), SimpleNamespace(index=1, name="Dev2", address="x"))

    def select_device(self, selector):
        self._called(f"select_device:{selector}")
        assert selector == 0
        return _direct_device()

    def list_channels(self):
        self._called("list_channels")
        return ("Dev1/AIN1", "Dev1/DIO1", "Dev1/AIN3")

    def get_device_attribute(self, device, attribute):
        self._called(f"get_attribute:{attribute}")
        assert device.name == "Dev1"
        return "VE3668N" if attribute == "Model" else self.machine_id

    def stream(self, **_kwargs):
        pytest.fail("discovery must never start capture")

    def acquire(self, **_kwargs):
        pytest.fail("discovery must never start capture")


def test_discovery_worker_runs_all_sdk_calls_on_its_qthread_and_never_captures():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    main_thread_id = threading.get_ident()
    client = _DiscoveryClient()
    results = []
    failures = []
    finished = []
    thread = visual.QThread()
    worker = visual.DeviceDiscoveryWorker(
        generation=7,
        client_factory=lambda: client,
    )
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.succeeded.connect(results.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(thread.quit, type=visual.Qt.DirectConnection)
    thread.finished.connect(lambda: finished.append(True))

    thread.start()
    _pump_qt_until(app, lambda: bool(finished))

    assert failures == []
    assert len(results) == 1
    result = results[0]
    assert result.generation == 7
    assert result.device.name == "Dev1"
    assert result.channel_routes == ("Dev1/AIN1", "Dev1/AIN3")
    assert result.device_snapshot["physical_channels"] == (0, 2)
    assert client.calls == [
        "enter",
        "list_devices",
        "select_device:0",
        "list_channels",
        "get_attribute:Model",
        "get_attribute:MachineId",
        "exit",
    ]
    assert client.worker_thread_ids
    assert set(client.worker_thread_ids) == {client.worker_thread_ids[0]}
    assert client.worker_thread_ids[0] != main_thread_id


def test_discovery_failure_becomes_unavailable_and_retry_needs_valid_snapshot():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    clients = [
        _DiscoveryClient(failure=RuntimeError("SDK unavailable")),
        _DiscoveryClient(),
    ]
    store = object()
    store_factory_calls = []

    def store_factory():
        store_factory_calls.append(True)
        return store

    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: clients.pop(0),
        calibration_store_factory=store_factory,
    )

    first_generation = coordinator.start_discovery()
    assert coordinator.state is recorder.DiscoveryState.DISCOVERING
    _pump_qt_until(
        app,
        lambda: coordinator.state is recorder.DiscoveryState.UNAVAILABLE
        and not coordinator.discovery_active,
    )
    assert coordinator.device_available is False
    assert coordinator.start_available is False
    assert "SDK unavailable" in coordinator.discovery_diagnostic

    second_generation = coordinator.retry_discovery()
    assert second_generation > first_generation
    assert coordinator.state is recorder.DiscoveryState.DISCOVERING
    _pump_qt_until(
        app,
        lambda: coordinator.state is recorder.DiscoveryState.IDLE
        and not coordinator.discovery_active,
    )
    assert coordinator.device_available is True
    assert coordinator.start_available is True
    assert coordinator.device_snapshot["physical_channels"] == (0, 2)
    assert coordinator.calibration_store is store
    assert store_factory_calls == [True]

    valid_snapshot = coordinator.device_snapshot
    coordinator.handle_discovery_failure(
        recorder.DeviceDiscoveryFailure(first_generation, "stale failure")
    )
    assert coordinator.state is recorder.DiscoveryState.IDLE
    assert coordinator.device_snapshot == valid_snapshot
    assert coordinator.discovery_diagnostic is None


def test_discovery_unexpected_exception_retires_thread_then_retry_reaches_idle():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    clients = [
        _DiscoveryClient(failure=TypeError("unexpected SDK result type")),
        _DiscoveryClient(),
    ]
    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: clients.pop(0),
        calibration_store=object(),
    )
    failure_diagnostics = []
    coordinator.discovery_failed.connect(failure_diagnostics.append)

    coordinator.start_discovery()
    _pump_qt_until(
        app,
        lambda: coordinator.state is recorder.DiscoveryState.UNAVAILABLE
        and not coordinator.discovery_active,
    )

    assert coordinator.device_available is False
    assert coordinator.discovery_diagnostic == "unexpected SDK result type"
    assert failure_diagnostics == ["unexpected SDK result type"]

    coordinator.retry_discovery()
    _pump_qt_until(
        app,
        lambda: coordinator.state is recorder.DiscoveryState.IDLE
        and not coordinator.discovery_active,
    )
    assert coordinator.device_available is True
    assert coordinator.device_snapshot["machine_id"] == "MID-001"
    assert failure_diagnostics == ["unexpected SDK result type"]


def test_discovery_failed_callback_can_retry_immediately_after_thread_retirement():
    visual = recorder._create_visual_types()
    app = visual.QApplication.instance() or visual.QApplication([])
    clients = [
        _DiscoveryClient(failure=TypeError("unexpected SDK result type")),
        _DiscoveryClient(),
    ]
    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: clients.pop(0),
        calibration_store=object(),
    )
    observed_terminal_states = []
    retry_errors = []
    retry_generations = []
    successes = []

    def observe_state(state):
        if state in (recorder.DiscoveryState.UNAVAILABLE, recorder.DiscoveryState.IDLE):
            observed_terminal_states.append((state, coordinator.discovery_active))

    def retry_from_failure(diagnostic):
        assert diagnostic == "unexpected SDK result type"
        assert coordinator.discovery_active is False
        try:
            retry_generations.append(coordinator.retry_discovery())
        except Exception as error:
            retry_errors.append(error)

    def observe_success(result):
        successes.append(result)
        assert coordinator.discovery_active is False

    coordinator.state_changed.connect(observe_state)
    coordinator.discovery_failed.connect(retry_from_failure)
    coordinator.discovery_succeeded.connect(observe_success)

    first_generation = coordinator.start_discovery()
    _pump_qt_until(
        app,
        lambda: bool(retry_errors)
        or (
            coordinator.state is recorder.DiscoveryState.IDLE
            and len(successes) == 1
        ),
    )

    assert retry_errors == []
    assert retry_generations == [first_generation + 1]
    assert observed_terminal_states == [
        (recorder.DiscoveryState.UNAVAILABLE, False),
        (recorder.DiscoveryState.IDLE, False),
    ]
    assert coordinator.device_available is True
    assert coordinator.device_snapshot["machine_id"] == "MID-001"


def test_discovery_stale_pending_outcome_is_discarded_at_thread_retirement():
    visual = recorder._create_visual_types()
    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: pytest.fail("discovery is not needed"),
        calibration_store=object(),
    )
    generation = coordinator.reserve_discovery_generation()
    coordinator._pending_discovery_outcome = recorder.DeviceDiscoveryFailure(
        generation - 1,
        "stale failure",
    )
    terminal_signals = []
    coordinator.discovery_failed.connect(terminal_signals.append)

    coordinator._handle_discovery_thread_finished()

    assert coordinator.state is recorder.DiscoveryState.DISCOVERING
    assert coordinator.discovery_diagnostic is None
    assert coordinator.device_available is False
    assert coordinator._pending_discovery_outcome is None
    assert terminal_signals == []


def test_frozen_metadata_preserves_physical_wav_order_and_missing_factors(tmp_path):
    from base.ve3668n_stores import VECalibrationStore

    snapshot = recorder.resolve_ve_device_snapshot(
        _direct_device(),
        model="VE3668N",
        machine_id="MID-001",
        routes=("Dev1/AIN3", "Dev1/AIN1"),
    )
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")

    frozen = recorder.build_frozen_ve_recording_metadata(snapshot, calibrations)
    metadata = frozen.to_dict()

    assert [
        (channel["wav_channel_index"], channel["physical_input_channel"])
        for channel in metadata["recorded_channels"]
    ] == [(0, 2), (1, 0)]
    assert [channel["factor_source"] for channel in metadata["recorded_channels"]] == [
        "none",
        "none",
    ]

    calibrations.save(
        snapshot,
        2,
        v2pa_factor=4.25,
        standard_spl=94.0,
        calibration_sample_rate=51200,
        calibration_duration_seconds=10.0,
        calibrated_at="2026-09-01T10:00:00+08:00",
    )
    snapshot["physical_channels"] = (0, 2)

    assert frozen.to_dict() == metadata


def test_corrupt_store_blocks_frozen_metadata_start_without_losing_device_state(
    tmp_path,
):
    from base.ve3668n_stores import VECalibrationStore

    path = tmp_path / "calibrations.json"
    path.write_text('{"schema_version": 1, "devices": {"MID-001": ', encoding="utf-8")
    calibrations = VECalibrationStore(path)
    visual = recorder._create_visual_types()
    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: pytest.fail("discovery is not needed"),
        calibration_store=calibrations,
    )
    generation = coordinator.reserve_discovery_generation()
    snapshot = recorder.resolve_ve_device_snapshot(
        _direct_device(),
        model="VE3668N",
        machine_id="MID-001",
        routes=("Dev1/AIN1",),
    )
    coordinator.handle_discovery_success(
        recorder.DeviceDiscoveryResult(
            generation=generation,
            device=_direct_device(),
            channel_routes=("Dev1/AIN1",),
            device_snapshot=snapshot,
        )
    )

    with pytest.raises(recorder.StartMetadataError, match="Invalid VE store"):
        coordinator.freeze_metadata_for_start()

    assert coordinator.state is recorder.DiscoveryState.IDLE
    assert coordinator.device_available is True
    assert coordinator.start_available is True
    assert "Invalid VE store" in coordinator.start_diagnostic


def test_frozen_metadata_uses_copy_on_read_authority_despite_external_mutation(
    tmp_path,
):
    from base.ve3668n_stores import VECalibrationStore

    visual = recorder._create_visual_types()
    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: pytest.fail("discovery is not needed"),
        calibration_store=VECalibrationStore(tmp_path / "calibrations.json"),
    )
    generation = coordinator.reserve_discovery_generation()
    snapshot = recorder.resolve_ve_device_snapshot(
        _direct_device(),
        model="VE3668N",
        machine_id="MID-001",
        routes=("Dev1/AIN3", "Dev1/AIN1"),
    )
    coordinator.handle_discovery_success(
        recorder.DeviceDiscoveryResult(
            generation=generation,
            device=_direct_device(),
            channel_routes=("Dev1/AIN3", "Dev1/AIN1"),
            device_snapshot=snapshot,
        )
    )

    exposed = coordinator.device_snapshot
    assert isinstance(exposed, dict)
    exposed["model"] = "attacker-model"
    exposed["machine_id"] = "attacker-machine"
    exposed["physical_channels"] = (0, 2)
    exposed["input_config"]["sample_rate"] = 48000
    replacement = dict(exposed)
    replacement["input_config"] = dict(exposed["input_config"])
    try:
        coordinator.device_snapshot = replacement
    except AttributeError:
        pass
    try:
        coordinator.channel_routes = ("Dev1/AIN1", "Dev1/AIN3")
    except AttributeError:
        pass

    current = coordinator.device_snapshot
    assert current["model"] == "VE3668N"
    assert current["machine_id"] == "MID-001"
    assert current["physical_channels"] == (2, 0)
    assert current["input_config"]["sample_rate"] == 51200
    assert coordinator.channel_routes == ("Dev1/AIN3", "Dev1/AIN1")

    metadata = coordinator.freeze_metadata_for_start().to_dict()
    assert metadata["acquisition"] == {
        "model": "VE3668N",
        "machine_id": "MID-001",
        "sample_rate": 51200,
        "input_mode": "IEPE",
        "unit": "V",
        "range_min": -10.0,
        "range_max": 10.0,
    }
    assert [
        channel["physical_input_channel"]
        for channel in metadata["recorded_channels"]
    ] == [2, 0]


@pytest.mark.parametrize("mismatch", ["sample_rate", "physical_order"])
def test_frozen_metadata_rejects_deliberate_internal_authority_mismatch(
    mismatch,
):
    from base.recording_process_protocol import FrozenConfig

    visual = recorder._create_visual_types()
    store_observations = []

    class TrackingCalibrationStore:
        def observe(self, device):
            store_observations.append(device)
            return {}

    coordinator = visual.DeviceDiscoveryCoordinator(
        client_factory=lambda: pytest.fail("discovery is not needed"),
        calibration_store=TrackingCalibrationStore(),
    )
    generation = coordinator.reserve_discovery_generation()
    snapshot = recorder.resolve_ve_device_snapshot(
        _direct_device(),
        model="VE3668N",
        machine_id="MID-001",
        routes=("Dev1/AIN3", "Dev1/AIN1"),
    )
    coordinator.handle_discovery_success(
        recorder.DeviceDiscoveryResult(
            generation=generation,
            device=_direct_device(),
            channel_routes=("Dev1/AIN3", "Dev1/AIN1"),
            device_snapshot=snapshot,
        )
    )
    corrupted = coordinator.device_snapshot
    if mismatch == "sample_rate":
        corrupted["input_config"]["sample_rate"] = 48000
    else:
        corrupted["physical_channels"] = (0, 2)
    coordinator._authoritative_discovery = replace(
        coordinator._authoritative_discovery,
        device_snapshot=FrozenConfig.snapshot(corrupted),
    )

    with pytest.raises(recorder.StartMetadataError, match="fixed|route order"):
        coordinator.freeze_metadata_for_start()

    assert store_observations == []


class FakeStream:
    def __init__(self, events, chunks):
        self.events = events
        self.chunks = iter(chunks)

    def __enter__(self):
        self.events.append("stream_enter")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.events.append("stream_exit")

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.chunks)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


class FakeClient:
    def __init__(
        self,
        events,
        *,
        devices=("device zero",),
        channels=("Dev1/AIN2", "Dev1/AIN1"),
        chunks=(),
    ):
        self.events = events
        self.devices = devices
        self.channels = channels
        self.chunks = chunks
        self.selected = None
        self.stream_arguments = None

    def __enter__(self):
        self.events.append("client_enter")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.events.append("client_exit")

    def list_devices(self):
        return self.devices

    def select_device(self, selector):
        self.selected = selector
        return self.devices[selector]

    def list_channels(self):
        return self.channels

    def stream(self, **arguments):
        self.stream_arguments = arguments
        return FakeStream(self.events, self.chunks)


class TrackingPublisher:
    def __init__(self, events, *, interrupt_on_call=None):
        self.events = events
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0
        self.saved = []

    def publish_pending(self, accumulator, *, sample_rate, timestamp):
        head = accumulator.peek_complete()
        if head is None:
            return None
        self.calls += 1
        self.events.append(f"publish_{self.calls}")
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        accumulator.acknowledge(head)
        path = Path(f"saved-{self.calls}.wav")
        self.saved.append(head.copy())
        return path


def chunk(*channels):
    return SimpleNamespace(samples=channels)


def install_small_accumulator(monkeypatch, events=None, *, interrupt_add=False):
    requested_sizes = []
    real_accumulator = recorder.SegmentAccumulator

    class SmallAccumulator(real_accumulator):
        def __init__(self, *, channel_count, frames_per_segment):
            requested_sizes.append(frames_per_segment)
            super().__init__(channel_count=channel_count, frames_per_segment=3)

        def add_channel_major(self, channel_major):
            super().add_channel_major(channel_major)
            if interrupt_add:
                if events is not None:
                    events.append("conversion_interrupt")
                raise KeyboardInterrupt

    monkeypatch.setattr(recorder, "SegmentAccumulator", SmallAccumulator)
    return requested_sizes


def _test_wav_metadata(channel_count=2):
    return {
        "recorded_channels": [
            {
                "wav_channel_index": index,
                "v2pa_factor": None,
                "standard_spl": None,
                "calibrated": False,
            }
            for index in range(channel_count)
        ]
    }


def test_streaming_run_publisher_writes_incrementally_and_publishes_metadata(tmp_path):
    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=2,
        timestamp=datetime(2026, 9, 1, 10, 20, 30),
    )

    assert publisher.temp_path.parent == tmp_path
    assert publisher.temp_path.name.startswith(".")
    assert publisher.write_channel_major(((1, 2), (11, 12))) == 2
    assert publisher.write_channel_major(((3,), (13,))) == 1

    metadata = _test_wav_metadata()
    path = publisher.publish(metadata=metadata)

    assert path == tmp_path / "recording_20260901_102030.wav"
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 51_200
    np.testing.assert_array_equal(samples, [[1, 11], [2, 12], [3, 13]])
    from base.wav_calibration_metadata import read_wav_calibration_metadata

    assert read_wav_calibration_metadata(path) == metadata
    assert publisher.accepted_frames == publisher.written_frames == 3
    assert not publisher.temp_path.exists()


def test_streaming_run_publisher_publishes_frozen_ve_metadata_after_json_readback(
    tmp_path,
):
    from base.recording_process_protocol import FrozenConfig
    from base.ve3668n_wav_metadata import validate_ve_wav_metadata
    from base.wav_calibration_metadata import read_wav_calibration_metadata
    from unit_test.base.ve3668n_fakes import wav_metadata

    frozen_metadata = FrozenConfig.snapshot(
        wav_metadata(("measured", "none"), sample_rate=51_200)
    )
    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=2,
        timestamp=datetime(2026, 9, 1, 10, 20, 30),
    )
    temp_path = publisher.temp_path
    publisher.write_channel_major(((1.0, 2.0), (11.0, 12.0)))

    path = publisher.publish(metadata=frozen_metadata)

    assert path == tmp_path / "recording_20260901_102030.wav"
    assert path.exists()
    assert not temp_path.exists()
    assert validate_ve_wav_metadata(read_wav_calibration_metadata(path)) == (
        validate_ve_wav_metadata(frozen_metadata)
    )


def test_streaming_run_publisher_rejects_genuinely_different_valid_ve_readback(
    tmp_path,
):
    from base.recording_process_protocol import FrozenConfig
    from base.wav_calibration_metadata import inspect_wav_calibration_metadata
    from unit_test.base.ve3668n_fakes import wav_metadata

    frozen_metadata = FrozenConfig.snapshot(
        wav_metadata(("measured", "none"), sample_rate=51_200)
    )

    def inspect_with_different_machine(path):
        result = inspect_wav_calibration_metadata(path)
        metadata = result.metadata
        metadata["acquisition"]["machine_id"] = "different-machine"
        return replace(result, metadata=metadata)

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=2,
        timestamp=datetime(2026, 9, 1, 10, 20, 30),
        metadata_reader=inspect_with_different_machine,
    )
    temp_path = publisher.temp_path
    publisher.write_channel_major(((1.0, 2.0), (11.0, 12.0)))

    with pytest.raises(OSError, match="WAV metadata readback did not match"):
        publisher.publish(metadata=frozen_metadata)

    assert not temp_path.exists()
    assert publisher.published_path is None
    assert list(tmp_path.glob("recording_*.wav")) == []


def test_streaming_run_publisher_selects_suffix_without_overwrite(tmp_path):
    timestamp = datetime(2026, 9, 1, 10, 20, 30)
    original = tmp_path / "recording_20260901_102030.wav"
    original.write_bytes(b"existing")
    publisher = recorder.StreamingRunPublisher(
        tmp_path, sample_rate=51_200, channel_count=1, timestamp=timestamp
    )
    publisher.write_channel_major(((1.0,),))

    path = publisher.publish(metadata=_test_wav_metadata(1))

    assert path == tmp_path / "recording_20260901_102030_1.wav"
    assert original.read_bytes() == b"existing"


class _FakeStreamingWriter:
    def __init__(self, path, **_kwargs):
        self.path = Path(path)
        self.parts = []
        self.closed = False
        self.path.write_bytes(b"temporary")

    def write(self, samples):
        self.parts.append(np.asarray(samples).copy())

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("subtype", "PCM_16", "FLOAT"),
        ("samplerate", 48_000, "sample rate"),
        ("channels", 1, "channel count"),
        ("frames", 1, "frame count"),
    ],
)
def test_streaming_run_publisher_rejects_each_header_mismatch(
    tmp_path, field, bad_value, message
):
    writer = None

    def writer_factory(path, **kwargs):
        nonlocal writer
        writer = _FakeStreamingWriter(path, **kwargs)
        return writer

    def header_reader(_path):
        values = dict(subtype="FLOAT", samplerate=51_200, channels=2, frames=2)
        values[field] = bad_value
        return SimpleNamespace(**values)

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=2,
        writer_factory=writer_factory,
        header_reader=header_reader,
        metadata_appender=lambda _path, _metadata: True,
        metadata_reader=lambda _path: _test_wav_metadata(),
    )
    publisher.write_channel_major(((1, 2), (11, 12)))

    with pytest.raises(OSError, match=message):
        publisher.publish(metadata=_test_wav_metadata())

    assert writer.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ["write", "metadata", "readback", "rename"])
def test_streaming_run_publisher_failure_discards_owned_temp(tmp_path, stage):
    writer = None

    class Writer(_FakeStreamingWriter):
        def write(self, samples):
            if stage == "write":
                raise OSError("write failed")
            super().write(samples)

    def writer_factory(path, **kwargs):
        nonlocal writer
        writer = Writer(path, **kwargs)
        return writer

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=2,
        writer_factory=writer_factory,
        header_reader=lambda _path: SimpleNamespace(
            subtype="FLOAT", samplerate=51_200, channels=2, frames=2
        ),
        metadata_appender=(
            (lambda _path, _metadata: False)
            if stage == "metadata"
            else (lambda _path, _metadata: True)
        ),
        metadata_reader=(
            (lambda _path: {"wrong": True})
            if stage == "readback"
            else (lambda _path: _test_wav_metadata())
        ),
        rename=(
            (lambda _source, _destination: (_ for _ in ()).throw(OSError("rename failed")))
            if stage == "rename"
            else os.rename
        ),
    )

    if stage == "write":
        with pytest.raises(OSError, match="write failed"):
            publisher.write_channel_major(((1, 2), (11, 12)))
        assert writer.closed
        assert list(tmp_path.iterdir()) == []
    else:
        publisher.write_channel_major(((1, 2), (11, 12)))
        with pytest.raises(OSError):
            publisher.publish(metadata=_test_wav_metadata())

    publisher.discard()
    assert writer.closed
    assert list(tmp_path.iterdir()) == []


def test_streaming_run_publisher_discard_is_idempotent(tmp_path):
    publisher = recorder.StreamingRunPublisher(
        tmp_path, sample_rate=51_200, channel_count=1
    )
    temp_path = publisher.temp_path

    publisher.discard()
    publisher.discard()

    assert not temp_path.exists()


def test_streaming_run_publisher_validates_header_before_metadata_and_rename(tmp_path):
    events = []

    class OrderedWriter(_FakeStreamingWriter):
        def close(self):
            events.append("writer_close")
            super().close()

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        writer_factory=lambda path, **kwargs: OrderedWriter(path, **kwargs),
        header_reader=lambda _path: (
            events.append("header_read")
            or SimpleNamespace(
                subtype="FLOAT", samplerate=51_200, channels=1, frames=1
            )
        ),
        metadata_appender=lambda _path, _metadata: (
            events.append("metadata_append") or True
        ),
        metadata_reader=lambda _path: (
            events.append("metadata_read") or _test_wav_metadata(1)
        ),
        rename=lambda source, destination: (
            events.append("rename") or os.rename(source, destination)
        ),
    )
    publisher.write_channel_major(((1.0,),))

    publisher.publish(metadata=_test_wav_metadata(1))

    assert events == [
        "writer_close",
        "header_read",
        "metadata_append",
        "metadata_read",
        "rename",
    ]


def test_streaming_run_publisher_close_failure_retains_temp_and_diagnostic(tmp_path):
    close_error = OSError("writer close failed")

    class CloseFailingWriter(_FakeStreamingWriter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            raise close_error

    writer = None

    def writer_factory(path, **kwargs):
        nonlocal writer
        writer = CloseFailingWriter(path, **kwargs)
        return writer

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        writer_factory=writer_factory,
    )
    publisher.write_channel_major(((1.0,),))
    temp_path = publisher.temp_path

    with pytest.raises(OSError, match="writer close failed") as first:
        publisher.discard()

    assert first.value is close_error
    assert publisher.close_error is close_error
    assert writer.close_calls == 1
    assert temp_path.exists()

    with pytest.raises(OSError, match="writer close failed") as second:
        publisher.discard()

    assert second.value is close_error
    assert writer.close_calls == 1
    assert temp_path.exists()


def test_streaming_run_publisher_preserves_primary_when_discard_cleanup_fails(
    tmp_path,
):
    primary = OSError("header inspection failed")
    cleanup = OSError("temporary unlink failed")
    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        header_reader=lambda _path: (_ for _ in ()).throw(primary),
        metadata_appender=lambda _path, _metadata: True,
        metadata_reader=lambda _path: _test_wav_metadata(1),
    )
    publisher.write_channel_major(((1.0,),))
    real_remove_temp = publisher._remove_temp
    publisher._remove_temp = lambda: (_ for _ in ()).throw(cleanup)

    with pytest.raises(OSError, match="header inspection failed") as caught:
        publisher.publish(metadata=_test_wav_metadata(1))

    assert caught.value is primary
    assert any("temporary WAV cleanup failed" in note for note in primary.__notes__)
    assert any("temporary unlink failed" in note for note in primary.__notes__)
    publisher._remove_temp = real_remove_temp
    publisher.discard()


@pytest.mark.parametrize("uncertain_stage", ["append", "readback"])
def test_streaming_run_publisher_default_metadata_ownership_failure_retains_files(
    tmp_path, monkeypatch, uncertain_stage
):
    from base import wav_calibration_metadata as metadata_io

    retained_handle = object()
    metadata_cleanup_path = tmp_path / f".{uncertain_stage}.metadata.tmp"
    metadata_cleanup_path.write_bytes(b"owned metadata temporary")
    close_detail = f"{uncertain_stage} metadata close failed"
    primary_detail = f"{uncertain_stage} metadata primary failure"
    calls = []

    def ownership_result(*, appended=None, metadata=None):
        return SimpleNamespace(
            appended=appended,
            status=metadata_io.WavCalibrationMetadataReadStatus.VALID,
            metadata=metadata,
            handles_released=False,
            cleanup_paths=(str(metadata_cleanup_path),),
            close_errors=(close_detail,),
            retained_handles=((uncertain_stage, retained_handle),),
            primary_error=primary_detail,
        )

    def append_result(_path, _metadata, logger=None):
        calls.append("append")
        if uncertain_stage == "append":
            return ownership_result(appended=True)
        return SimpleNamespace(
            appended=True,
            handles_released=True,
            cleanup_paths=(),
            close_errors=(),
            retained_handles=(),
            primary_error=None,
        )

    def inspect_result(_path, logger=None):
        calls.append("readback")
        return ownership_result(metadata=_test_wav_metadata(1))

    monkeypatch.setattr(
        metadata_io, "append_wav_calibration_metadata_result", append_result
    )
    monkeypatch.setattr(metadata_io, "inspect_wav_calibration_metadata", inspect_result)
    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        timestamp=datetime(2026, 9, 1, 10, 20, 30),
    )
    publisher.write_channel_major(((1.0,),))
    wav_temp_path = publisher.temp_path

    with pytest.raises(OSError) as caught:
        publisher.publish(metadata=_test_wav_metadata(1))

    assert primary_detail in str(caught.value)
    assert close_detail in str(caught.value)
    assert calls == (["append"] if uncertain_stage == "append" else ["append", "readback"])
    assert publisher.published_path is None
    assert wav_temp_path.exists()
    assert metadata_cleanup_path.exists()
    assert publisher.metadata_ownership_uncertain
    assert publisher.metadata_retained_handles == ((uncertain_stage, retained_handle),)
    assert publisher.metadata_cleanup_paths == (str(metadata_cleanup_path),)
    assert publisher.metadata_close_errors == (close_detail,)
    assert publisher.metadata_primary_errors == (primary_detail,)
    assert not list(tmp_path.glob("recording_*.wav"))


def test_streaming_run_publisher_rename_collision_race_reuses_completed_temp(
    tmp_path,
):
    timestamp = datetime(2026, 9, 1, 10, 20, 30)
    first_candidate = tmp_path / "recording_20260901_102030.wav"
    rename_calls = []

    def colliding_rename(source, destination):
        rename_calls.append((Path(source), Path(destination)))
        if len(rename_calls) == 1:
            destination.write_bytes(b"concurrent winner")
            raise FileExistsError("destination appeared concurrently")
        os.rename(source, destination)

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        timestamp=timestamp,
        rename=colliding_rename,
    )
    publisher.write_channel_major(((1.0, 2.0),))
    original_temp = publisher.temp_path

    path = publisher.publish(metadata=_test_wav_metadata(1))

    assert path == tmp_path / "recording_20260901_102030_1.wav"
    assert first_candidate.read_bytes() == b"concurrent winner"
    assert len(rename_calls) == 2
    assert rename_calls[0][0] == rename_calls[1][0] == original_temp
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 51_200
    np.testing.assert_array_equal(samples[:, 0], [1.0, 2.0])


@pytest.mark.parametrize(
    "post_commit_error",
    [OSError("rename reported failure after commit"), KeyboardInterrupt()],
)
def test_streaming_run_publisher_reconciles_exception_after_committed_rename(
    tmp_path, post_commit_error
):
    rename_calls = []

    def rename_then_raise(source, destination):
        rename_calls.append((Path(source), Path(destination)))
        os.rename(source, destination)
        raise post_commit_error

    publisher = recorder.StreamingRunPublisher(
        tmp_path,
        sample_rate=51_200,
        channel_count=1,
        timestamp=datetime(2026, 9, 1, 10, 20, 30),
        rename=rename_then_raise,
    )
    publisher.write_channel_major(((1.0, 2.0),))
    temp_path = publisher.temp_path

    path = publisher.publish(metadata=_test_wav_metadata(1))

    assert path == tmp_path / "recording_20260901_102030.wav"
    assert publisher.published_path == path
    assert path.exists()
    assert not temp_path.exists()
    assert len(rename_calls) == 1
    assert list(tmp_path.glob("recording_*.wav")) == [path]
    assert publisher.publish(metadata=_test_wav_metadata(1)) == path
    assert len(rename_calls) == 1


@pytest.mark.parametrize("interrupt_point", ["path_conversion", "descriptor_close"])
def test_streaming_run_publisher_initialization_interrupt_cleans_raw_temp_ownership(
    tmp_path, monkeypatch, interrupt_point
):
    created = {}
    close_attempts = []
    successful_closes = []
    real_mkstemp = tempfile.mkstemp
    real_close = os.close

    def tracking_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.update(descriptor=descriptor, name=name)
        return descriptor, name

    real_path = recorder.Path

    def interrupting_path(value):
        if interrupt_point == "path_conversion" and value == created.get("name"):
            raise KeyboardInterrupt
        return real_path(value)

    close_interrupted = False

    def interrupting_close(descriptor):
        nonlocal close_interrupted
        close_attempts.append(descriptor)
        if (
            interrupt_point == "descriptor_close"
            and descriptor == created.get("descriptor")
            and not close_interrupted
        ):
            close_interrupted = True
            raise KeyboardInterrupt
        real_close(descriptor)
        successful_closes.append(descriptor)

    monkeypatch.setattr(recorder.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(recorder, "Path", interrupting_path)
    monkeypatch.setattr(recorder.os, "close", interrupting_close)

    with pytest.raises(KeyboardInterrupt):
        recorder.StreamingRunPublisher(
            tmp_path,
            sample_rate=51_200,
            channel_count=1,
        )

    descriptor = created["descriptor"]
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert successful_closes == [descriptor]
    assert close_attempts.count(descriptor) == (
        2 if interrupt_point == "descriptor_close" else 1
    )
    assert not Path(created["name"]).exists()
    assert list(tmp_path.iterdir()) == []


class _VisualPublisher:
    def __init__(self, *, fail_write=False):
        self.fail_write = fail_write
        self.blocks = []
        self.publish_calls = 0
        self.discard_calls = 0
        self.written_frames = 0

    def write_channel_major(self, samples):
        if self.fail_write:
            raise OSError("writer failed")
        block = tuple(np.asarray(channel, dtype=np.float32).copy() for channel in samples)
        self.blocks.append(block)
        frame_count = len(block[0])
        self.written_frames += frame_count
        return frame_count

    def publish(self, *, metadata):
        self.publish_calls += 1
        self.metadata = metadata
        return Path("published.wav")

    def discard(self):
        self.discard_calls += 1


class _VisualClient(FakeClient):
    def __init__(self, chunks, **kwargs):
        super().__init__([], chunks=chunks, **kwargs)


def _run_visual(
    client,
    publisher,
    *,
    target_frames=4,
    sink=None,
    controller=None,
    device_snapshot=None,
    accepted_frame_sink=None,
):
    return recorder.run_visual_recording(
        lambda: client,
        device_selector=0,
        channel_routes=("Dev1/AIN2", "Dev1/AIN1"),
        target_frames=target_frames,
        metadata={"run": "frozen"},
        publisher=publisher,
        waveform_sink=sink,
        stop_controller=controller or recorder.StopController(),
        device_snapshot=device_snapshot,
        accepted_frame_sink=accepted_frame_sink,
    )


def test_visual_recording_accepts_exact_five_minute_target_boundary():
    controller = recorder.StopController()
    controller.request_stop()
    client = _VisualClient(())
    publisher = _VisualPublisher()

    result = recorder.run_visual_recording(
        lambda: client,
        device_selector=0,
        channel_routes=("Dev1/AIN2", "Dev1/AIN1"),
        target_frames=recorder.SAMPLE_RATE * recorder.SEGMENT_SECONDS,
        metadata={"run": "frozen"},
        publisher=publisher,
        waveform_sink=None,
        stop_controller=controller,
    )

    assert result.no_audio
    assert result.failure is None
    assert publisher.discard_calls == 1


def test_visual_recording_rejects_over_five_minute_target_before_resource_use():
    client_factory_calls = []
    publisher = _VisualPublisher()

    with pytest.raises(ValueError, match="five-minute maximum"):
        recorder.run_visual_recording(
            lambda: client_factory_calls.append("client") or _VisualClient(()),
            device_selector=0,
            channel_routes=("Dev1/AIN2", "Dev1/AIN1"),
            target_frames=(recorder.SAMPLE_RATE * recorder.SEGMENT_SECONDS) + 1,
            metadata={"run": "frozen"},
            publisher=publisher,
            waveform_sink=None,
            stop_controller=recorder.StopController(),
        )

    assert client_factory_calls == []
    assert publisher.publish_calls == 0
    assert publisher.discard_calls == 0


def test_visual_recording_revalidates_routes_streams_raw_volts_and_slices_target_frames():
    client = _VisualClient((chunk((1, 2, 3, 4, 5), (11, 12, 13, 14, 15)),))
    publisher = _VisualPublisher()
    previews = []
    sink = SimpleNamespace(append=lambda samples: previews.append(samples))

    result = _run_visual(client, publisher, target_frames=3, sink=sink)

    assert result.failure is None
    assert result.path == Path("published.wav")
    assert result.accepted_frames == result.written_frames == 3
    assert client.selected == 0
    assert client.stream_arguments == {
        "channels": ("Dev1/AIN2", "Dev1/AIN1"),
        "mode": "iepe_voltage",
        "sample_rate": 51_200,
        "samples_per_chunk": 5_120,
        "min_value": -10.0,
        "max_value": 10.0,
        "terminal": "single_ended",
        "timeout": 2.0,
        "sensitivity": 1000.0,
    }
    np.testing.assert_array_equal(publisher.blocks[0][0], [1, 2, 3])
    np.testing.assert_array_equal(previews[0][1], [11, 12, 13])
    assert publisher.publish_calls == 1


@pytest.mark.parametrize(
    "fresh_routes",
    [
        ("Dev1/AIN1",),
        ("Dev1/AIN1", "Dev1/AIN2"),
        ("Dev1/AIN2", "Dev1/AIN1", "Dev1/AIN3"),
    ],
)
def test_visual_recording_rejects_any_ordered_ain_route_snapshot_change(fresh_routes):
    client = _VisualClient((), channels=fresh_routes)
    publisher = _VisualPublisher()

    result = _run_visual(client, publisher)

    assert "route snapshot changed" in result.failure.message
    assert client.stream_arguments is None
    assert publisher.discard_calls == 1


def test_visual_recording_route_revalidation_skips_non_ain_entries():
    client = _VisualClient(
        (chunk((1, 2), (11, 12)),),
        channels=("Dev1/AIN2", "Dev1/status", "Dev1/AIN1"),
    )
    publisher = _VisualPublisher()

    result = _run_visual(client, publisher, target_frames=2)

    assert result.failure is None
    assert result.accepted_frames == 2


def test_visual_recording_manual_stop_publishes_admitted_partial_data():
    controller = recorder.StopController()

    class StopSink:
        def append(self, _samples):
            controller.request_stop()

    publisher = _VisualPublisher()
    result = _run_visual(
        _VisualClient((chunk((1, 2), (11, 12)), chunk((3, 4), (13, 14)))),
        publisher,
        target_frames=10,
        sink=StopSink(),
        controller=controller,
    )

    assert result.path == Path("published.wav")
    assert result.accepted_frames == result.written_frames == 2
    assert publisher.publish_calls == 1


def test_visual_recording_zero_frame_stop_discards_and_returns_no_audio():
    controller = recorder.StopController()
    controller.request_stop()
    publisher = _VisualPublisher()

    result = _run_visual(
        _VisualClient((chunk((1,), (11,)),)), publisher, controller=controller
    )

    assert result.path is None
    assert result.no_audio
    assert result.accepted_frames == result.written_frames == 0
    assert publisher.publish_calls == 0
    assert publisher.discard_calls == 1


def test_visual_recording_target_stop_race_has_one_terminal_publication():
    controller = recorder.StopController()

    class RacingSink:
        def append(self, _samples):
            controller.request_stop()

    publisher = _VisualPublisher()
    result = _run_visual(
        _VisualClient((chunk((1, 2, 3), (11, 12, 13)),)),
        publisher,
        target_frames=2,
        sink=RacingSink(),
        controller=controller,
    )

    assert result.accepted_frames == 2
    assert publisher.publish_calls == 1
    assert publisher.discard_calls == 0


def test_visual_recording_preview_failure_warns_once_and_keeps_valid_wav():
    class FailingSink:
        def __init__(self):
            self.calls = 0

        def append(self, _samples):
            self.calls += 1
            raise RuntimeError("render failed")

    sink = FailingSink()
    publisher = _VisualPublisher()
    result = _run_visual(
        _VisualClient((chunk((1, 2), (11, 12)), chunk((3, 4), (13, 14)))),
        publisher,
        sink=sink,
    )

    assert result.path == Path("published.wav")
    assert result.accepted_frames == result.written_frames == 4
    assert result.preview_warning == "Waveform preview disabled: render failed"
    assert result.failure is None
    assert sink.calls == 1


def test_visual_recording_preview_failure_does_not_suppress_later_progress():
    class FailingSink:
        def append(self, _samples):
            raise RuntimeError("render failed")

    progress = []
    result = _run_visual(
        _VisualClient((chunk((1, 2), (11, 12)), chunk((3, 4), (13, 14)))),
        _VisualPublisher(),
        sink=FailingSink(),
        accepted_frame_sink=progress.append,
    )

    assert result.failure is None
    assert result.accepted_frames == 4
    assert progress == [2, 4]
    assert result.preview_warning == "Waveform preview disabled: render failed"


@pytest.mark.parametrize(
    ("replacement", "expected_detail"),
    [
        ({"name": "Other"}, "name"),
        ({"address": "192.0.2.99"}, "address"),
        ({"model": "VE3668N-clone"}, "Model"),
        ({"machine_id": "MID-REPLACED"}, "MachineId"),
    ],
)
def test_visual_recording_rejects_same_index_identity_replacement_before_stream(
    replacement, expected_detail
):
    snapshot = {
        "backend": "vkinging",
        "model": "VE3668N",
        "machine_id": "MID-001",
        "name": "Dev1",
        "address": "192.0.2.1",
        "physical_channels": (1, 0),
        "max_input_channels": 2,
        "available": True,
        "input_config": {
            "sample_rate": 51_200,
            "input_mode": "IEPE",
            "unit": "V",
            "range_min": -10.0,
            "range_max": 10.0,
        },
    }

    class IdentityClient(_VisualClient):
        def __init__(self):
            super().__init__((chunk((1,), (11,)),))
            self.identity = {
                "name": "Dev1",
                "address": "192.0.2.1",
                "model": "VE3668N",
                "machine_id": "MID-001",
            }
            self.identity.update(replacement)
            self.devices = (
                SimpleNamespace(
                    index=0,
                    name=self.identity["name"],
                    address=self.identity["address"],
                ),
            )

        def get_device_attribute(self, _device, attribute):
            return self.identity["model" if attribute == "Model" else "machine_id"]

    client = IdentityClient()
    publisher = _VisualPublisher()
    result = _run_visual(client, publisher, device_snapshot=snapshot, target_frames=1)

    assert result.path is None
    assert result.failure is not None
    assert expected_detail in result.failure.message
    assert client.stream_arguments is None
    assert publisher.publish_calls == 0
    assert publisher.discard_calls == 1


@pytest.mark.parametrize("failure_kind", ["capture", "writer"])
def test_visual_recording_capture_or_writer_failure_exposes_no_path(failure_kind):
    chunks = (OSError("capture failed"),) if failure_kind == "capture" else (chunk((1,), (11,)),)
    publisher = _VisualPublisher(fail_write=failure_kind == "writer")

    result = _run_visual(_VisualClient(chunks), publisher)

    assert result.path is None
    assert result.failure is not None
    assert not result.ownership_uncertain
    if failure_kind == "writer":
        assert result.accepted_frames == 1
        assert result.written_frames == 0
    assert publisher.publish_calls == 0
    assert publisher.discard_calls == 1


def test_visual_recording_short_write_reports_truthful_accepted_and_written_counts():
    class ShortWriter(_VisualPublisher):
        def write_channel_major(self, samples):
            super().write_channel_major(samples)
            return 1

    publisher = ShortWriter()

    result = _run_visual(
        _VisualClient((chunk((1, 2), (11, 12)),)), publisher, target_frames=2
    )

    assert result.path is None
    assert result.failure is not None
    assert "admitted frame count" in result.failure.message
    assert result.accepted_frames == 2
    assert result.written_frames == 1
    assert publisher.publish_calls == 0


@pytest.mark.parametrize("returned_count", [-1, True, 1.5, 3])
def test_visual_recording_invalid_writer_count_does_not_invent_written_frames(
    returned_count,
):
    class InvalidCountWriter(_VisualPublisher):
        def write_channel_major(self, samples):
            super().write_channel_major(samples)
            return returned_count

    result = _run_visual(
        _VisualClient((chunk((1, 2), (11, 12)),)),
        InvalidCountWriter(),
        target_frames=2,
    )

    assert result.failure is not None
    assert "invalid frame count" in result.failure.message
    assert result.accepted_frames == 2
    assert result.written_frames == 0


def test_visual_recording_cleanup_failure_marks_ownership_uncertain_and_does_not_publish():
    class OwnershipError(OSError):
        ownership_uncertain = True

    class UncertainClient(_VisualClient):
        def __exit__(self, exception_type, exception, traceback):
            super().__exit__(exception_type, exception, traceback)
            raise OwnershipError("native ClearTask failed")

    publisher = _VisualPublisher()
    result = _run_visual(
        UncertainClient((chunk((1, 2), (11, 12)),)), publisher, target_frames=2
    )

    assert result.path is None
    assert result.failure is not None
    assert result.failure.ownership_uncertain
    assert result.ownership_uncertain
    assert publisher.publish_calls == 0
    assert publisher.discard_calls == 1


def test_visual_recording_attached_native_cleanup_diagnostic_marks_ownership_uncertain():
    capture_error = OSError("capture failed")
    capture_error.add_note(
        "Task cleanup also failed: VkDaqStopTask failed with SDK result -9"
    )
    publisher = _VisualPublisher()

    result = _run_visual(_VisualClient((capture_error,)), publisher)

    assert result.failure.ownership_uncertain
    assert result.ownership_uncertain
    assert result.path is None
    assert "capture failed" in result.failure.message
    assert "Task cleanup also failed" in result.failure.message
    assert "VkDaqStopTask failed with SDK result -9" in result.failure.message
    assert result.failure.message.count("capture failed") == 1


def test_accumulator_splits_uneven_channel_major_chunks_without_loss():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    accumulator.add_channel_major(((1, 2, 3), (11, 12, 13)))
    accumulator.add_channel_major(((4, 5, 6, 7), (14, 15, 16, 17)))

    assert accumulator.peek_complete().dtype == np.float32
    np.testing.assert_array_equal(
        accumulator.peek_complete(),
        [[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],
    )
    first = accumulator.peek_complete()
    accumulator.acknowledge(first)
    accumulator.queue_tail()
    np.testing.assert_array_equal(accumulator.peek_complete(), [[6, 16], [7, 17]])


def test_accumulator_queues_multiple_segments_from_oversized_chunk_and_keeps_tail():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=3)

    accumulator.add_channel_major((range(1, 9), range(11, 19)))

    first = accumulator.peek_complete()
    np.testing.assert_array_equal(first, [[1, 11], [2, 12], [3, 13]])
    accumulator.acknowledge(first)
    second = accumulator.peek_complete()
    np.testing.assert_array_equal(second, [[4, 14], [5, 15], [6, 16]])
    accumulator.acknowledge(second)
    accumulator.queue_tail()
    np.testing.assert_array_equal(accumulator.peek_complete(), [[7, 17], [8, 18]])


def test_accumulator_rejects_mismatched_channel_lengths():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    with pytest.raises(ValueError, match="same number of frames"):
        accumulator.add_channel_major(((1, 2), (11,)))


def test_accumulator_rejects_wrong_channel_count():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    with pytest.raises(ValueError, match="expected 2 channels"):
        accumulator.add_channel_major(((1, 2),))


def test_accumulator_acknowledge_rejects_non_head_object():
    accumulator = SegmentAccumulator(channel_count=1, frames_per_segment=2)
    accumulator.add_channel_major(((1, 2, 3, 4),))
    head = accumulator.peek_complete()
    equal_but_distinct = head.copy()

    with pytest.raises(ValueError, match="head"):
        accumulator.acknowledge(equal_but_distinct)

    assert accumulator.peek_complete() is head


def test_accumulator_empty_queue_has_no_tail_segment():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    assert accumulator.queue_tail() is None
    assert accumulator.peek_complete() is None


class InterruptingAccumulator(SegmentAccumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupt_phase = None

    def _commit_state(self, next_state):
        if self.interrupt_phase == "before":
            raise KeyboardInterrupt
        super()._commit_state(next_state)
        if self.interrupt_phase == "after":
            raise KeyboardInterrupt


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_add_interrupt_exposes_only_old_or_complete_new_state(
    interrupt_phase,
):
    accumulator = InterruptingAccumulator(channel_count=2, frames_per_segment=3)
    accumulator.add_channel_major(((1,), (11,)))
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.add_channel_major(((2, 3, 4), (12, 13, 14)))

    accumulator.interrupt_phase = None
    accumulator.queue_tail()
    if interrupt_phase == "before":
        np.testing.assert_array_equal(accumulator.peek_complete(), [[1, 11]])
    else:
        first = accumulator.peek_complete()
        np.testing.assert_array_equal(first, [[1, 11], [2, 12], [3, 13]])
        accumulator.acknowledge(first)
        np.testing.assert_array_equal(accumulator.peek_complete(), [[4, 14]])


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_acknowledge_interrupt_is_transactional(interrupt_phase):
    accumulator = InterruptingAccumulator(channel_count=1, frames_per_segment=2)
    accumulator.add_channel_major(((1, 2, 3, 4),))
    first = accumulator.peek_complete()
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.acknowledge(first)

    if interrupt_phase == "before":
        assert accumulator.peek_complete() is first
    else:
        np.testing.assert_array_equal(accumulator.peek_complete(), [[3], [4]])


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_queue_tail_interrupt_never_loses_or_duplicates_tail(
    interrupt_phase,
):
    accumulator = InterruptingAccumulator(channel_count=1, frames_per_segment=3)
    accumulator.add_channel_major(((1, 2),))
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.queue_tail()

    accumulator.interrupt_phase = None
    if interrupt_phase == "before":
        assert accumulator.peek_complete() is None
        accumulator.queue_tail()
    else:
        assert accumulator.queue_tail() is None
    np.testing.assert_array_equal(accumulator.peek_complete(), [[1], [2]])


@pytest.mark.parametrize("channel_count", [0, -1, 1.5, True])
def test_accumulator_requires_positive_integer_channel_count(channel_count):
    with pytest.raises(ValueError, match="channel_count"):
        SegmentAccumulator(channel_count=channel_count, frames_per_segment=5)


@pytest.mark.parametrize("frames_per_segment", [0, -1, 1.5, True])
def test_accumulator_requires_positive_integer_segment_size(frames_per_segment):
    with pytest.raises(ValueError, match="frames_per_segment"):
        SegmentAccumulator(channel_count=2, frames_per_segment=frames_per_segment)


def pending_accumulator():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=3)
    accumulator.add_channel_major(((1.25, 2.5, 3.75), (-1.25, -2.5, -3.75)))
    return accumulator


def test_wav_publisher_writes_float32_multichannel_samples_at_requested_rate(tmp_path):
    accumulator = pending_accumulator()
    publisher = WavPublisher(tmp_path)

    published = publisher.publish_pending(
        accumulator,
        sample_rate=48_000,
        timestamp=datetime(2026, 8, 25, 12, 34, 56),
    )

    assert published == tmp_path / "recording_20260825_123456.wav"
    rate, samples = wavfile.read(published)
    assert rate == 48_000
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(
        samples,
        [[1.25, -1.25], [2.5, -2.5], [3.75, -3.75]],
    )
    assert accumulator.peek_complete() is None


def test_wav_publisher_selects_incrementing_suffix_without_overwriting(tmp_path):
    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    original = tmp_path / "recording_20260825_123456.wav"
    first_collision = tmp_path / "recording_20260825_123456_1.wav"
    original.write_bytes(b"original recording")
    first_collision.write_bytes(b"first collision")

    published = WavPublisher(tmp_path).publish_pending(
        pending_accumulator(), sample_rate=48_000, timestamp=timestamp
    )

    assert published == tmp_path / "recording_20260825_123456_2.wav"
    assert original.read_bytes() == b"original recording"
    assert first_collision.read_bytes() == b"first collision"


def test_publish_collision_during_rename_reuses_completed_temp_wav(tmp_path):
    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    first_candidate = tmp_path / "recording_20260825_123456.wav"
    write_calls = []
    rename_calls = []

    def tracking_writer(path, rate, samples):
        write_calls.append((path, rate, samples.copy()))
        wavfile.write(path, rate, samples)

    def colliding_rename(source, destination):
        rename_calls.append((source, destination))
        if len(rename_calls) == 1:
            destination.write_bytes(b"concurrent winner")
            raise FileExistsError("destination appeared concurrently")
        os.rename(source, destination)

    accumulator = pending_accumulator()
    published = WavPublisher(
        tmp_path, wav_writer=tracking_writer, rename=colliding_rename
    ).publish_pending(accumulator, sample_rate=48_000, timestamp=timestamp)

    assert published == tmp_path / "recording_20260825_123456_1.wav"
    assert first_candidate.read_bytes() == b"concurrent winner"
    assert len(write_calls) == 1
    assert len(rename_calls) == 2
    assert rename_calls[0][0] == rename_calls[1][0]
    rate, samples = wavfile.read(published)
    assert rate == 48_000
    np.testing.assert_array_equal(samples, write_calls[0][2])
    assert accumulator.peek_complete() is None


@pytest.mark.parametrize("failure_stage", ["write", "rename"])
def test_wav_publish_failure_cleans_temp_and_retains_pending_head(
    tmp_path, failure_stage
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()

    def failing_writer(path, rate, samples):
        if failure_stage == "write":
            raise OSError("write failed")
        wavfile.write(path, rate, samples)

    def failing_rename(source, destination):
        raise OSError("rename failed")

    publisher = WavPublisher(
        tmp_path,
        wav_writer=failing_writer,
        rename=failing_rename if failure_stage == "rename" else os.rename,
    )

    with pytest.raises(OSError, match=f"{failure_stage} failed"):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    assert accumulator.peek_complete() is head
    assert list(tmp_path.iterdir()) == []


def test_interrupt_after_successful_rename_acknowledges_once_without_republish(tmp_path):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()
    acknowledge_calls = []
    real_acknowledge = accumulator.acknowledge

    def tracking_acknowledge(segment):
        acknowledge_calls.append(segment)
        real_acknowledge(segment)

    accumulator.acknowledge = tracking_acknowledge

    def rename_then_interrupt(source, destination):
        os.rename(source, destination)
        raise KeyboardInterrupt

    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    publisher = WavPublisher(tmp_path, rename=rename_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator, sample_rate=48_000, timestamp=timestamp
        )

    final_path = tmp_path / "recording_20260825_123456.wav"
    assert final_path.exists()
    assert acknowledge_calls == [head]
    assert accumulator.peek_complete() is None
    assert (
        WavPublisher(tmp_path).publish_pending(
            accumulator, sample_rate=48_000, timestamp=timestamp
        )
        is None
    )
    assert list(tmp_path.iterdir()) == [final_path]


@pytest.mark.parametrize("interrupt_stage", ["write", "rename"])
def test_interrupt_before_successful_rename_cleans_temp_and_retains_head(
    tmp_path, interrupt_stage
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()

    def interrupt_during_write(path, rate, samples):
        raise KeyboardInterrupt

    def interrupt_before_rename(source, destination):
        raise KeyboardInterrupt

    publisher = WavPublisher(
        tmp_path,
        wav_writer=interrupt_during_write if interrupt_stage == "write" else None,
        rename=interrupt_before_rename if interrupt_stage == "rename" else None,
    )

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    assert accumulator.peek_complete() is head
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("interrupt_point", ["path_conversion", "descriptor_close"])
def test_interrupt_after_mkstemp_closes_descriptor_removes_temp_and_retains_head(
    tmp_path, monkeypatch, interrupt_point
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()
    publisher = WavPublisher(tmp_path)
    created = {}
    real_mkstemp = tempfile.mkstemp
    real_close = os.close

    def tracking_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.update(descriptor=descriptor, name=name)
        return descriptor, name

    def interrupt_during_path_conversion(value):
        if interrupt_point == "path_conversion" and value == created.get("name"):
            raise KeyboardInterrupt
        return Path(value)

    close_interrupted = False

    def interrupt_during_first_close(descriptor):
        nonlocal close_interrupted
        if (
            interrupt_point == "descriptor_close"
            and descriptor == created.get("descriptor")
            and not close_interrupted
        ):
            close_interrupted = True
            raise KeyboardInterrupt
        real_close(descriptor)

    monkeypatch.setattr(recorder.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(recorder, "Path", interrupt_during_path_conversion)
    monkeypatch.setattr(recorder.os, "close", interrupt_during_first_close)

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    descriptor = created["descriptor"]
    temp_name = created["name"]
    try:
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert not Path(temp_name).exists()
        assert accumulator.peek_complete() is head
    finally:
        try:
            real_close(descriptor)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)


def test_recorder_selects_first_device_all_channels_and_exact_stream_settings(
    monkeypatch, capsys
):
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))
    publisher = TrackingPublisher(events)
    requested_sizes = install_small_accumulator(monkeypatch)

    status = recorder.run_recorder(lambda: client, publisher=publisher)

    assert status == 0
    assert client.selected == 0
    assert client.stream_arguments == {
        "channels": ("Dev1/AIN2", "Dev1/AIN1"),
        "mode": "iepe_voltage",
        "sample_rate": 51_200,
        "samples_per_chunk": 51_200,
        "min_value": -10.0,
        "max_value": 10.0,
        "terminal": "single_ended",
        "timeout": 2.0,
        "sensitivity": 1000.0,
    }
    assert client.stream_arguments["timeout"] > 0
    assert requested_sizes == [51_200 * 300]
    assert requested_sizes == [15_360_000]
    output = capsys.readouterr().out
    assert "Selected device: device zero" in output
    assert "Selected channels: Dev1/AIN2, Dev1/AIN1" in output
    assert "Recording started" in output
    assert "Recording stopped" in output


@pytest.mark.parametrize(
    ("devices", "channels", "message"),
    [
        ((), ("Dev1/AIN1",), "No Vkinging devices found"),
        (("device zero",), (), "No channels found"),
    ],
)
def test_recorder_rejects_missing_device_or_channel_before_stream_creation(
    devices, channels, message
):
    events = []
    client = FakeClient(events, devices=devices, channels=channels)

    with pytest.raises(RuntimeError, match=message):
        recorder.run_recorder(lambda: client, publisher=TrackingPublisher(events))

    assert client.stream_arguments is None


def test_recorder_publishes_complete_segments_while_one_stream_stays_open(
    monkeypatch, capsys
):
    events = []
    client = FakeClient(
        events,
        chunks=(
            chunk((1, 2, 3, 4, 5, 6), (11, 12, 13, 14, 15, 16)),
            KeyboardInterrupt(),
        ),
    )
    publisher = TrackingPublisher(events)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    assert events == [
        "client_enter",
        "stream_enter",
        "publish_1",
        "publish_2",
        "stream_exit",
        "client_exit",
    ]
    assert len(publisher.saved) == 2
    output = capsys.readouterr().out
    assert output.count("Saved: saved-1.wav") == 1
    assert output.count("Saved: saved-2.wav") == 1


@pytest.mark.parametrize("interrupt_stage", ["iteration", "conversion", "publication"])
def test_first_interrupt_closes_native_contexts_before_finalizing_complete_and_tail(
    monkeypatch, interrupt_stage
):
    events = []
    chunks = [chunk((1, 2, 3, 4), (11, 12, 13, 14))]
    interrupt_add = interrupt_stage == "conversion"
    if interrupt_stage == "iteration":
        chunks.append(KeyboardInterrupt())
    elif interrupt_stage == "publication":
        chunks.append(KeyboardInterrupt())
    publisher = TrackingPublisher(
        events, interrupt_on_call=1 if interrupt_stage == "publication" else None
    )
    client = FakeClient(events, chunks=chunks)
    install_small_accumulator(
        monkeypatch, events=events, interrupt_add=interrupt_add
    )

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    stream_exit = events.index("stream_exit")
    client_exit = events.index("client_exit")
    finalization_publications = [
        index
        for index, event in enumerate(events)
        if event.startswith("publish_") and index > client_exit
    ]
    assert stream_exit < client_exit < min(finalization_publications)
    assert [saved.shape[0] for saved in publisher.saved] == [3, 1]


def test_recorder_reports_post_rename_interrupt_path_once_then_finalizes(
    monkeypatch, tmp_path, capsys
):
    events = []
    client = FakeClient(
        events,
        chunks=(chunk((1, 2, 3, 4), (11, 12, 13, 14)),),
    )
    destinations = []

    def interrupt_after_first_rename(source, destination):
        os.rename(source, destination)
        destinations.append(destination)
        events.append(f"rename_{len(destinations)}")
        if len(destinations) == 1:
            raise KeyboardInterrupt

    publisher = WavPublisher(tmp_path, rename=interrupt_after_first_rename)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    first_path = destinations[0]
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines.count(f"Saved: {first_path}") == 1
    assert len(destinations) == 2
    assert all(path.exists() for path in destinations)
    assert events.index("rename_1") < events.index("stream_exit")
    assert events.index("client_exit") < events.index("rename_2")


def test_recorder_does_not_publish_empty_tail(monkeypatch):
    events = []
    client = FakeClient(
        events,
        chunks=(chunk((1, 2, 3), (11, 12, 13)), KeyboardInterrupt()),
    )
    publisher = TrackingPublisher(events)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    assert publisher.calls == 1


def test_second_interrupt_during_finalization_cleans_temp_and_returns_130(
    tmp_path, capsys
):
    events = []
    client = FakeClient(
        events,
        channels=("Dev1/AIN1",),
        chunks=(chunk((1, 2, 3)), KeyboardInterrupt()),
    )

    def interrupting_writer(path, rate, samples):
        raise KeyboardInterrupt

    publisher = WavPublisher(tmp_path, wav_writer=interrupting_writer)

    status = recorder.run_recorder(lambda: client, publisher=publisher)

    assert status == 130
    assert events[-2:] == ["stream_exit", "client_exit"]
    assert list(tmp_path.iterdir()) == []
    assert "buffered data was not saved" in capsys.readouterr().err.lower()


def test_injected_client_factory_keeps_vendor_binding_lazy(monkeypatch):
    monkeypatch.delitem(sys.modules, "vkinging_daq", raising=False)
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))

    assert recorder.run_recorder(
        lambda: client, publisher=TrackingPublisher(events)
    ) == 0
    assert "vkinging_daq" not in sys.modules


def test_recorder_imports_vendor_client_only_when_factory_is_omitted(monkeypatch):
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))
    calls = []

    def vendor_factory():
        calls.append("constructed")
        return client

    fake_binding = SimpleNamespace(VkDaqClient=vendor_factory)
    monkeypatch.setitem(sys.modules, "vkinging_daq", fake_binding)

    assert recorder.run_recorder(publisher=TrackingPublisher(events)) == 0
    assert calls == ["constructed"]


def test_main_prints_one_concise_cli_error_and_returns_nonzero(monkeypatch, capsys):
    def fail():
        raise OSError("SDK unavailable")

    monkeypatch.setattr(recorder, "run_recorder", fail)

    assert recorder.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("SDK unavailable") == 1
