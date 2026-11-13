"""Exercise both application entry paths without a native recording device."""

import json
from datetime import date, datetime
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5 import QtWidgets, sip
from PyQt5.QtCore import QObject, pyqtSignal

import main_window
from base import hardware_selection, recording_service
from base.db_manager import DataSave
from base.recording_process_protocol import FrozenConfig, SoundcardPrepareResult
from base.recording_service import RecordingWorkerStatus
from consts import error_code
from main_window_Launcher import MainWindowLauncher
from ui import recording_service_bridge, vkinging_hardware_controls
from ui.video_controller import VideoController
from unit_test.base.vkinging_fakes import device_info, input_config


class RecordingBridge(QObject):
    worker_status_changed = pyqtSignal(object)
    shutting_down = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.service = SimpleNamespace(closed=Event(), can_start_recording=True)
        self.worker_status = RecordingWorkerStatus(0, "idle")
        self.started = False
        self.hardware_busy = False
        self.probes = []
        self.prewarms = []
        self.retained_ve_signature = None
        self.shutdown_calls = 0

    def ensure_resident(self):
        self.started = True
        return "accepted"

    def prepare_soundcard(self, request, callback):
        assert self.started
        self.probes.append((request, callback))
        return "accepted"

    def prewarm_ve(self, request, callback):
        assert self.started
        self.prewarms.append((request, callback))
        return "accepted"

    def release_ve(self, signature, callback):
        if signature != self.retained_ve_signature:
            self.retained_ve_signature = None
        callback("released", ())
        return "released"

    def change_worker(self, generation, state="ready"):
        self.worker_status = RecordingWorkerStatus(generation, state)
        self.worker_status_changed.emit(self.worker_status)

    def answer_probe(self, index=-1, *, name="Default", generation=1):
        request, callback = self.probes[index]
        if name is None:
            callback(SoundcardPrepareResult(request.probe_id, generation, detail="input identity is missing"))
            return
        device = FrozenConfig.snapshot({"backend": "sounddevice", "index": 7, "name": name,
                                        "hostapi": 0, "hostapi_name": "WASAPI", "max_input_channels": 3})
        callback(SoundcardPrepareResult(request.probe_id, generation, device, (0, 1, 2), 48000))

    def finish_prewarm(self, *, success=True):
        request, callback = self.prewarms[-1]
        if success:
            self.retained_ve_signature = request.signature
        callback(SimpleNamespace(warmup_id=request.warmup_id, signature=request.signature,
                                 success=success, stage="completed" if success else "start_task",
                                 code=None, detail="", diagnostics=(), ownership_safe=True))

    def shutdown(self, callback=None):
        if not self.service.closed.is_set():
            self.shutdown_calls += 1
            self.shutting_down.emit()
            self.service.closed.set()
        if callback is not None:
            callback()


class Discovery:
    def __init__(self, *, on_result):
        self._on_result = on_result
        self.starts = 0
        self.generation = 0

    def start(self):
        self.starts += 1
        self.generation += 1
        return self.generation

    refresh = start

    def cancel(self):
        self.generation += 1

    def close(self):
        pass

    def answer(self, *devices, generation=None):
        self._on_result(SimpleNamespace(
            generation=self.generation if generation is None else generation,
            handles_released=True, result=SimpleNamespace(devices=devices, diagnostics=())))


class AnalysisServiceStub:
    """Keep hardware/startup tests independent of resident analysis processes."""

    def __init__(self):
        self.sync_calls = []
        self.shutdown_calls = 0
        self.poll_calls = 0
        self.closed = False
        self.allow_close = True
        self.events = []
        self.station_states = {}
        self.pending_task_count = 0
        self.submissions = []

    def submit_batch(self, payload):
        self.submissions.append(payload)
        return {"accepted": True, "accepted_task_ids": [t["task_id"] for t in payload["tasks"]], "issues": []}

    def sync_stations(self, station_ids):
        self.sync_calls.append(tuple(station_ids))
        self.station_states = {sid: {"state": "ready", "pid": index + 1}
                               for index, sid in enumerate(station_ids)}
        return {"accepted": True, "error": None}

    def shutdown(self):
        self.shutdown_calls += 1

    def poll_events(self):
        self.poll_calls += 1
        events, self.events = self.events, []
        if self.shutdown_calls and self.allow_close and not self.closed:
            self.closed = True
            events.append({"event_type": "service_closed", "payload": {}})
        return tuple(events)


@pytest.fixture
def startup(ui_qapp, monkeypatch, tmp_path):
    path = tmp_path / "hardware.json"
    database_path = tmp_path / "audio_data.db"
    database = DataSave(str(database_path))
    try:
        code, detail = database.create_table()
        assert code == error_code.OK, detail
    finally:
        database.close()
    monkeypatch.setattr(main_window, "DATABASE_PATH", str(database_path))
    bridge = RecordingBridge()
    mic = {"name": "Default", "index": 0, "hostapi": 0, "hostapi_name": "WASAPI", "max_input_channels": 3}
    manager = hardware_selection.SoundDeviceManager
    monkeypatch.setattr(ui_qapp, "_ve_prewarm_lifetime", None, raising=False)
    monkeypatch.setattr(ui_qapp, "_recording_bridge", None, raising=False)
    monkeypatch.setattr(hardware_selection, "_HARDWARE_SELECTION_PATH", str(path))
    monkeypatch.setattr(manager, "refresh_available_device", lambda: None)
    monkeypatch.setattr(manager, "get_device_info", lambda: {"WASAPI": {"input": [mic]}})
    monkeypatch.setattr(manager, "get_api_info", lambda *_: {"name": "WASAPI"})
    monkeypatch.setattr(manager, "change_default_input_device", lambda *_: None)
    monkeypatch.setattr(hardware_selection, "_os_default_device", lambda kind: mic if kind == "mic" else None)
    monkeypatch.setattr(vkinging_hardware_controls, "DiscoveryService", Discovery)
    monkeypatch.setattr(VideoController, "start_preview", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "get_current_version", lambda self: "test")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", Mock())
    windows = []
    state = SimpleNamespace(analysis_service=AnalysisServiceStub())

    class Window(main_window.MainWindow):
        def __init__(self, **kwargs):
            windows.append(self)
            kwargs.setdefault("analysis_service", state.analysis_service)
            super().__init__(**kwargs)

    monkeypatch.setattr(main_window, "MainWindow", Window)

    def login(execute):
        monkeypatch.setattr(main_window, "LoginWindow", lambda: SimpleNamespace(on_exec=lambda: execute(windows[-1])))

    def launch(execute, *, launcher=False, **options):
        login(execute)
        if not launcher:
            return Window(recording_bridge=bridge, **options)
        monkeypatch.setattr(QtWidgets, "QApplication", lambda _args: ui_qapp)
        monkeypatch.setattr(recording_service, "RecordingService", lambda: bridge.service)
        monkeypatch.setattr(recording_service_bridge, "RecordingServiceBridge", lambda *_: bridge)

        def fail_launch(_parent, _title, detail):
            pytest.fail(detail)

        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", fail_launch)
        application = MainWindowLauncher()
        assert bridge.started
        assert bridge.probes == []
        application.on_loader_finished()
        assert application.window.recording_bridge is bridge
        return application.window

    state.launch, state.login, state.path, state.bridge, state.mic = launch, login, path, bridge, mic
    state.database_path = database_path
    yield state
    for window in windows:
        if not sip.isdeleted(window):
            window.close()
            ui_qapp.processEvents()
            sip.delete(window)


@pytest.mark.parametrize("launcher", [False, True])
def test_soundcard_preparation_can_finish_inside_first_login(startup, launcher):
    bridge = startup.bridge

    def login(window):
        assert bridge.started
        assert len(bridge.probes) == 1
        request, _ = bridge.probes[0]
        assert request.name is request.hostapi_name is None
        device = window.workbench_recording_device
        assert device.snapshot is None
        assert not device.login_ready
        bridge.change_worker(1)
        bridge.answer_probe()
        assert device.snapshot.channels == (0, 1, 2)
        assert device.snapshot.sample_rate == 48000
        assert not device.login_ready
        return "Engineer", "test"

    window = startup.launch(login, launcher=launcher)
    device = window.workbench_recording_device
    assert device.login_ready
    assert device.snapshot.device["index"] == 7
    assert window.mic_channels == [0]


@pytest.mark.parametrize("saved_day, expected", [(date(2026, 11, 12), "10000"), (date(2026, 11, 11), "1")])
def test_daily_round_is_loaded_before_login_without_migrating_sample(startup, monkeypatch, saved_day, expected):
    from base.load_config import LoadUiConfig
    from base.workbench_round_store import RoundSnapshot, WorkbenchRoundStore
    from ui.sequence import motor_workbench_controller as controller_module

    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(now=lambda: datetime(2026, 11, 12, 9)))
    monkeypatch.setattr(LoadUiConfig, "load_last_recorded_info", staticmethod(
        lambda _logger: {"sample_number": "007", "current_test_round": 777},
    ))
    WorkbenchRoundStore().save(RoundSnapshot(saved_day, 10000))

    def login(window):
        sequence = window.sequence_window
        assert sequence.toolsbar.sample_number_lineedit.text() == expected
        assert sequence.toolsbar.sample_number_lineedit.isReadOnly()
        assert not sequence.player_btn.isEnabled()
        sequence.toolsbar.current_round_spinbox.setValue(9999)
        assert sequence.toolsbar.sample_number_lineedit.text() == expected
        return "Engineer", "test"

    startup.launch(login)
    assert WorkbenchRoundStore().load() == RoundSnapshot(date(2026, 11, 12), int(expected))


def test_saved_missing_soundcard_never_uses_legacy_fallback_and_can_be_reselected(startup, monkeypatch):
    startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "Missing", "mic_channels": [2]}))
    bridge = startup.bridge
    window = startup.launch(lambda _: ("Engineer", "test"))
    bridge.change_worker(1)
    request, _ = bridge.probes[0]
    assert (request.name, request.hostapi_name) == ("Missing", "WASAPI")
    assert window.mic == startup.mic and window.mic_channels == [0]
    bridge.answer_probe(name=None)
    device = window.workbench_recording_device
    assert device.snapshot is None and "重新选择" in device.detail

    monkeypatch.setattr(main_window, "open_hardware_selection_window", lambda **_: (False, None, [], None, []))
    window.on_hardware_window_init()
    assert len(bridge.probes) == 1
    assert "重新选择" in device.detail

    def select(**_):
        hardware_selection.save_if_changed(startup.mic, None, [1], [], strict=True)
        startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "External", "mic_channels": [2]}))
        return True, None, [], startup.mic, [1]

    monkeypatch.setattr(main_window, "open_hardware_selection_window", select)
    window.on_hardware_window_init()
    assert device.snapshot is None
    assert bridge.probes[-1][0].name == "Default"
    bridge.answer_probe(0, name="Missing")
    assert device.snapshot is None
    bridge.answer_probe()
    assert device.snapshot.channels == (0, 1, 2)
    assert window.mic_channels == [1]

    refresh = window.sequence_window.refresh_channel_windows

    def refresh_after_save():
        assert device.snapshot is None
        refresh()

    monkeypatch.setattr(window.sequence_window, "refresh_channel_windows", refresh_after_save)
    window.on_hardware_window_init()
    assert len(bridge.probes) == 3
    bridge.answer_probe()
    assert device.snapshot.channels == (0, 1, 2)


def test_startup_probe_uses_same_saved_identity_as_legacy_restore(startup, monkeypatch):
    startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "Default", "mic_channels": [2]}))
    enumerate_devices = hardware_selection._enumerate_devices

    def replace_saved_selection():
        startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "External", "mic_channels": [0]}))
        return enumerate_devices()

    monkeypatch.setattr(hardware_selection, "_enumerate_devices", replace_saved_selection)
    window = startup.launch(lambda _: ("Engineer", "test"))
    request, _ = startup.bridge.probes[0]
    assert window.mic == startup.mic and window.mic_channels == [2]
    assert (request.name, request.hostapi_name) == (window.mic["name"], window.mic["hostapi_name"])


def test_injected_bridge_shuts_down_once_when_application_quits(startup, ui_qapp, monkeypatch):
    shutdown = Mock(wraps=startup.bridge.shutdown)
    monkeypatch.setattr(startup.bridge, "shutdown", shutdown)
    startup.launch(lambda _: (None, None))
    startup.launch(lambda _: (None, None))

    try:
        ui_qapp.aboutToQuit.emit()
        shutdown.assert_called_once_with()
    finally:
        ui_qapp.aboutToQuit.disconnect(shutdown)


def test_login_cancel_and_account_switch_preserve_resident_device(startup):
    bridge = startup.bridge
    window = startup.launch(lambda _: (None, None))
    bridge.change_worker(1)
    bridge.answer_probe()
    device = window.workbench_recording_device
    prepared = device.snapshot
    assert prepared is not None and not device.login_ready

    def switch(window):
        assert not device.login_ready
        assert device.snapshot is prepared
        return "Engineer", "different account"

    startup.login(switch)
    window.on_login_window_init()
    assert device.login_ready and window.user_name == "different account"
    startup.login(lambda _: (None, None) if not device.login_ready else pytest.fail("still admitted"))
    window.on_login_window_init()
    assert device.login_ready and window.user_name == "different account"
    assert device.snapshot is prepared and len(bridge.probes) == 1


def test_retired_worker_and_shutdown_reject_late_soundcard_results(startup, ui_qapp):
    bridge = startup.bridge
    window = startup.launch(lambda _: ("Engineer", "test"))
    device = window.workbench_recording_device
    bridge.change_worker(1)
    bridge.answer_probe()
    assert device.snapshot.generation == 1
    bridge.change_worker(1, "unavailable")
    assert device.snapshot is None
    bridge.change_worker(2, "starting")
    bridge.change_worker(2)
    assert len(bridge.probes) == 2
    bridge.answer_probe(0)
    assert device.snapshot is None
    bridge.answer_probe(generation=2)
    assert device.snapshot.generation == 2

    bridge.change_worker(3)
    assert len(bridge.probes) == 3
    window.close()
    ui_qapp.processEvents()
    assert bridge.shutdown_calls == 1
    bridge.answer_probe(generation=3)
    assert device.snapshot is None and not device.login_ready


def test_ve_prewarms_discovered_full_inventory_at_profile_rate_before_login(startup, ui_qapp):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    inventory = device_info(physical_channels=[7, 1, 4])
    profile = input_config(44100, range_min=-0.5, range_max=0.5)
    profiles = SimpleNamespace(load=lambda _device, _calibrations: profile)

    def login(window):
        device = window.workbench_recording_device
        assert bridge.started and window.ve_discovery.service.starts == 1
        assert not bridge.probes
        window.sequence_window.sequence_config = [{"seq1": {"acq": {"detail": {"sample_rate": 96000}}}}]
        bridge.change_worker(1)
        window.ve_discovery.service.answer(inventory)
        ui_qapp.processEvents()
        request, _ = bridge.prewarms[0]
        assert request.channels == (7, 1, 4)
        assert request.sample_rate == 44100
        assert request.signature[-2:] == (-0.5, 0.5)
        assert device.snapshot is None
        bridge.finish_prewarm()
        assert device.snapshot.ve_signature == request.signature
        assert device.snapshot.physical_to_column == {7: 0, 1: 1, 4: 2}
        assert not device.login_ready
        return "Engineer", "test"

    window = startup.launch(login, ve_profile_store=profiles)
    device = window.workbench_recording_device
    prepared = device.snapshot
    assert device.login_ready and window.mic_channels == [1]
    assert window._ve_signature(window.mic, window.mic_channels)[2:4] == ((1,), 96000)

    # Legacy calibration can leave a subset in the worker. The capability
    # snapshot still supplies the full signature to recording admission.
    bridge.retained_ve_signature = window._ve_signature(window.mic, window.mic_channels)
    assert device.snapshot is prepared and bridge.retained_ve_signature != prepared.ve_signature
    bridge.retained_ve_signature = prepared.ve_signature
    assert device.snapshot is prepared

    bridge.change_worker(1, "unavailable")
    assert device.snapshot is None
    bridge.change_worker(2)
    assert window.ve_discovery.service.starts == 2
    assert device.snapshot is None
    window.ve_discovery.service.answer(inventory)
    ui_qapp.processEvents()
    assert device.snapshot.generation == 2
    assert device.snapshot.ve_signature == prepared.ve_signature
    assert len(bridge.prewarms) == 1


def test_cancel_hardware_settings_resumes_only_pending_ve_discovery(startup, ui_qapp, monkeypatch):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    saved_selection = startup.path.read_bytes()
    inventory = device_info()
    profiles = SimpleNamespace(load=lambda device, _calibrations: device["input_config"])
    window = startup.launch(lambda _: ("Engineer", "test"), ve_profile_store=profiles)
    discovery = window.ve_discovery.service
    device = window.workbench_recording_device
    bridge.change_worker(1)
    discovery.answer(inventory)
    ui_qapp.processEvents()
    bridge.finish_prewarm()
    assert device.snapshot.generation == 1

    monkeypatch.setattr(main_window, "open_hardware_selection_window", lambda **_: (False, None, [], None, []))
    selected = window.mic
    window.on_hardware_window_init()
    assert discovery.starts == 1
    assert window.mic is selected

    bridge.change_worker(1, "unavailable")
    bridge.change_worker(2)
    assert discovery.starts == 2 and device.snapshot is None
    interrupted_generation = discovery.generation
    assert window.mic["available"]

    window.on_hardware_window_init()
    assert discovery.starts == 3
    assert window.mic is selected and startup.path.read_bytes() == saved_selection
    discovery.answer(inventory, generation=interrupted_generation)
    ui_qapp.processEvents()
    assert device.snapshot is None
    discovery.answer(inventory)
    ui_qapp.processEvents()
    assert device.snapshot.generation == 2
    assert len(bridge.prewarms) == 1


def test_accepting_new_ve_selection_ignores_scan_started_inside_hardware_dialog(startup, ui_qapp, monkeypatch):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    inventory = device_info()
    replacement = device_info(machine_id="new-machine", physical_channels=[6, 2])
    profiles = SimpleNamespace(load=lambda device, _calibrations: device["input_config"])
    window = startup.launch(lambda _: ("Engineer", "test"), ve_profile_store=profiles)
    discovery = window.ve_discovery.service
    device = window.workbench_recording_device
    bridge.change_worker(1)
    discovery.answer(inventory)
    ui_qapp.processEvents()
    bridge.finish_prewarm()
    assert device.snapshot.device["machine_id"] == inventory["machine_id"]

    def save_replacement(**options):
        assert options["mic_device"]["machine_id"] == inventory["machine_id"]
        bridge.change_worker(1, "unavailable")
        bridge.change_worker(2)
        assert discovery.starts == 2 and device.snapshot is None
        discovery.answer(inventory)
        return True, None, [], replacement, [2]

    monkeypatch.setattr(main_window, "open_hardware_selection_window", save_replacement)
    window.on_hardware_window_init()
    prepared = device.snapshot
    assert prepared.generation == 2 and prepared.channels == (6, 2)
    assert prepared.device["machine_id"] == replacement["machine_id"]

    ui_qapp.processEvents()
    assert device.snapshot is prepared
    assert window.mic is replacement
    assert discovery.starts == 2 and len(bridge.prewarms) == 1
