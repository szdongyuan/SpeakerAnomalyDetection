"""Communication restarts must not discard a partially completed product round."""

from copy import deepcopy
from threading import Thread

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QObject, pyqtSignal

from base.hardware_trigger import serial_discrete_input_worker as worker_module
from base.load_config import LoadUiConfig
from base.unified_hid_device_manager import UnifiedHardwareManager
from unit_test.test_serial_product_ports import A, B, IDLE, PortHost


class FakeWorker(QObject):
    sig_state_changed = pyqtSignal(object)
    sig_status = pyqtSignal(object)

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = deepcopy(config)
        self.running = False

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.running = False
        self.sig_status.emit({
            "running": False, "connected": False, "message": "listener stopped",
        })

    def opened(self):
        self.sig_status.emit({
            "running": True, "connected": True, "error": "", "message": "opened",
        })

    def failed(self):
        self.running = False
        self.sig_status.emit({
            "running": False, "connected": False,
            "error": "port unavailable", "message": "port unavailable",
        })

    def frame(self, frame):
        self.sig_state_changed.emit({"raw_hex": frame, "product_full_frame": True})


@pytest.fixture
def connected_round(ui_qapp, monkeypatch):
    monkeypatch.setattr(worker_module, "SerialDiscreteInputWorker", FakeWorker)
    managers = []

    def create(*ports, idle="", polling=False):
        host = PortHost(*ports, idle=idle)
        config = LoadUiConfig.normalize_serial_discrete_input_config({
            "enabled": True,
            "serial_settings": {"port": "FAKE"},
            "polling_settings": {"enabled": polling},
            "port_switch_idle_code": idle,
        })
        host._serial_trigger_config = config
        host._show_serial_product_error_once = lambda detail: None
        manager = UnifiedHardwareManager()
        host.hw_manager = manager
        managers.append(manager)
        manager.sig_serial_trigger_status.connect(host.on_serial_trigger_status_changed)
        manager.sig_serial_full_frame.connect(host.on_serial_full_frame_received)
        assert host._start_serial_product_listener(config)["ok"]
        manager.serial_worker.opened()
        manager.serial_worker.frame(A)
        host.complete_current()
        host._manual_product_condition_results["group_1:condition_1"] = "OK"
        return host, manager

    yield create
    for manager in managers:
        manager.sig_serial_trigger_status.disconnect()
        manager.sig_serial_full_frame.disconnect()
        manager.stop_serial_discrete_input_listener()
    ui_qapp.processEvents()


def progress(host):
    return deepcopy({
        "round": host._manual_product_condition_group_id,
        "completed": host._manual_product_condition_completed_keys,
        "results": host._manual_product_condition_results,
        "port": getattr(host, "_serial_product_port_index", 0),
        "gear": host._manual_product_condition_index,
        "latch": host._serial_product_latched_frame,
        "waiting_idle": getattr(host, "_serial_product_waiting_port_idle", False),
        "waiting_close": host._serial_product_waiting_for_close,
        "pending_close": host._serial_product_pending_close_frame,
    })


def reconfigure(host, field, value):
    config = deepcopy(host._serial_trigger_config)
    config["polling_settings"][field] = value
    return host._start_serial_product_listener(config)


@pytest.mark.parametrize("polling,field,value", [
    (False, "enabled", True), (True, "enabled", False), (True, "interval_ms", 1200),
])
def test_mode_and_interval_restart_preserve_round_and_continue_next_gear(
    connected_round, polling, field, value,
):
    host, manager = connected_round((A, B), (A, B), polling=polling)
    before = progress(host)
    old_worker = manager.serial_worker
    assert reconfigure(host, field, value)["ok"]
    assert manager.serial_worker is not old_worker
    assert progress(host) == before
    assert host.discarded_groups == []
    manager.serial_worker.opened()
    assert progress(host) == before
    manager.serial_worker.frame(A)
    assert len(host.started) == 1
    manager.serial_worker.frame(B)
    assert host.started == ["group_1:condition_1", "group_1:condition_2"]
    assert host._manual_product_condition_group_id == before["round"]


@pytest.mark.parametrize("idle", ["", IDLE])
def test_port_boundary_survives_reconfiguration(connected_round, idle):
    host, manager = connected_round((A,), ((A if idle else B),), idle=idle)
    before = progress(host)
    assert reconfigure(host, "enabled", True)["ok"]
    manager.serial_worker.opened()
    assert progress(host) == before
    manager.serial_worker.frame(A)
    assert len(host.started) == 1
    if idle:
        manager.serial_worker.frame(IDLE)
    manager.serial_worker.frame(A if idle else B)
    assert host.started == ["group_1:condition_1", "group_2:condition_1"]
    assert host._manual_product_condition_group_id == before["round"]


@pytest.mark.parametrize("field,value", [
    ("bytesize", 7), ("parity", "E"), ("stopbits", 2), ("timeout", 0.3),
])
def test_serial_parameters_restart_with_new_values_and_keep_progress(
    connected_round, field, value,
):
    host, manager = connected_round((A, B))
    before = progress(host)
    old_worker = manager.serial_worker
    config = deepcopy(host._serial_trigger_config)
    config["serial_settings"][field] = value
    assert host._start_serial_product_listener(config)["ok"]
    assert manager.serial_worker is not old_worker
    assert not old_worker.isRunning()
    assert manager.serial_worker.config["serial_settings"][field] == value
    manager.serial_worker.opened()
    assert progress(host) == before
    assert host.discarded_groups == []
    manager.serial_worker.frame(B)
    assert host._manual_product_condition_group_id == before["round"]
    assert host.started[-1] == "group_1:condition_2"


@pytest.mark.parametrize("old_running", [False, True])
@pytest.mark.parametrize("delete_old", [False, True])
def test_queued_old_worker_frames_and_status_cannot_touch_new_listener(
    connected_round, ui_qapp, old_running, delete_old,
):
    host, manager = connected_round((A, B))
    before = progress(host)
    old_worker = manager.serial_worker

    def emit_delayed():
        old_worker.frame(B)
        old_worker.sig_status.emit({
            "running": False, "connected": False, "error": "old worker closed",
        })
        old_worker.running = old_running

    emitter = Thread(target=emit_delayed)
    emitter.start()
    emitter.join(timeout=2)
    assert not emitter.is_alive()
    # Leave the old signals in Qt's event queue while replacing the listener.
    assert reconfigure(host, "enabled", True)["ok"]
    manager.serial_worker.opened()
    if delete_old:
        sip.delete(old_worker)
    ui_qapp.processEvents()
    assert progress(host) == before
    assert len(host.started) == 1
    assert manager.get_serial_discrete_input_status()["connected"]
    assert host.discarded_groups == []


def test_restart_open_failure_retains_results_until_successful_retry(connected_round):
    host, manager = connected_round((A, B))
    before = progress(host)
    assert reconfigure(host, "enabled", True)["ok"]
    manager.serial_worker.failed()
    status = manager.get_serial_discrete_input_status()
    assert status["error"] == "port unavailable" and not status["connected"]
    assert progress(host) == before
    assert host.discarded_groups == []
    assert host._start_serial_product_listener(host._serial_trigger_config)["ok"]
    assert not manager.get_serial_discrete_input_status()["error"]
    manager.serial_worker.opened()
    manager.serial_worker.frame(B)
    assert host._manual_product_condition_group_id == before["round"]
    assert host.started[-1] == "group_1:condition_2"


def test_real_disconnect_after_restart_still_uses_existing_abort_rule(connected_round):
    host, manager = connected_round((A, B))
    group = host._manual_product_condition_group_id
    assert reconfigure(host, "enabled", True)["ok"]
    assert host.discarded_groups == []
    manager.serial_worker.opened()
    manager.serial_worker.failed()
    assert host.discarded_groups == [(group, True)]
    assert host._manual_product_condition_group_id == ""


def test_connection_preview_and_failed_restore_preserve_round_after_dialog_closes(
    connected_round, monkeypatch,
):
    host, manager = connected_round((A, B))
    before = progress(host)
    host._serial_trigger_config_dialog_open = True
    monkeypatch.setattr(manager, "test_serial_discrete_input_connection", lambda config: {
        "ok": True, "raw_hex": A, "message": "test passed",
    })
    assert host._test_serial_trigger_connection(host._serial_trigger_config)["connected"]
    host._serial_trigger_config_dialog_open = False
    manager.serial_worker.failed()
    assert progress(host) == before
    assert host.discarded_groups == []
