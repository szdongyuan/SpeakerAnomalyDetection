from copy import deepcopy
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from base.hardware_trigger import serial_discrete_input_worker as worker_module
from base.hardware_trigger.serial_polling import parse_polling_settings
from base.load_config import LoadUiConfig
from base.unified_hid_device_manager import UnifiedHardwareManager
from unit_test.test_serial_product_condition_runtime import _Logger
from unit_test.test_serial_product_ports import A, B, IDLE, PortHost


QUERY = "FE 02 00 00 00 04 6D C6"


def configuration(enabled=False):
    return LoadUiConfig.normalize_serial_discrete_input_config({
        "enabled": True,
        "serial_settings": {"port": "FAKE"},
        "polling_settings": {"enabled": enabled, "interval_ms": 1, "query_command_hex": QUERY},
    })


@pytest.mark.parametrize("settings", [
    {"enabled": True, "query_command_hex": ""},
    {"enabled": True, "query_command_hex": "01 0"},
    {"enabled": True, "query_command_hex": "XX"},
    {"enabled": "false"},
    {"interval_ms": 0}, {"interval_ms": -1},
    {"interval_ms": float("nan")}, {"interval_ms": float("inf")},
    {"interval_ms": True}, {"interval_ms": "50"}, None,
    {"interval_ms": 1201}, {"interval_ms": 1500},
    {"interval_ms": 50.5}, {"interval_ms": 50.0},
])
def test_invalid_polling_settings(settings):
    with pytest.raises(ValueError):
        parse_polling_settings(settings)


def test_missing_switch_is_passive_even_with_old_query():
    assert parse_polling_settings({"query_command_hex": QUERY}) == (b"", 0.05)
    assert parse_polling_settings({"enabled": False, "query_command_hex": "old invalid value"}) == (b"", 0.05)


@pytest.mark.parametrize("interval", [1, 50, 1200])
def test_interval_boundaries(interval):
    assert parse_polling_settings({"interval_ms": interval}) == (b"", interval / 1000.0)


@pytest.mark.parametrize("polling", [False, True])
@pytest.mark.parametrize("idle", ["", IDLE])
def test_transport_triggers_all_ordered_product_gears(monkeypatch, polling, idle):
    host = PortHost((A, B), (A, B), idle=idle)
    config = configuration(polling)
    config["port_switch_idle_code"] = idle
    host._serial_trigger_config = config
    monkeypatch.setattr(worker_module.LogManager, "set_log_handler", lambda name: _Logger())
    worker = worker_module.SerialDiscreteInputWorker(config, host._serial_full_frame_candidates())
    # Includes an early gear and a duplicate at the port boundary.
    frames = [B, A, B, B] + ([IDLE] if idle else []) + [A, B]

    class Port:
        is_open = True

        def __init__(self):
            self.writes = []
            self.reply_ready = not polling

        def reset_input_buffer(self):
            pass

        def write(self, data):
            self.writes.append(data)
            assert data == bytes.fromhex(QUERY)
            self.reply_ready = True

        @property
        def in_waiting(self):
            return len(bytes.fromhex(frames[0])) if self.reply_ready and frames else 0

        def read(self, size):
            data = bytes.fromhex(frames.pop(0))
            self.reply_ready = not polling
            if not frames:
                worker._is_running = False
            return data

        def close(self):
            self.is_open = False

    port = Port()
    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **kwargs: port))
    cycles = []

    def tick(seconds):
        cycles.append(seconds)
        if len(cycles) > 20:
            worker._is_running = False

    monkeypatch.setattr(worker._stop_event, "wait", tick)
    manager = UnifiedHardwareManager()
    manager.serial_config = config
    manager.serial_worker = worker
    worker.sig_state_changed.connect(manager._on_serial_state_changed)

    def process(payload):
        previous = len(host.started)
        host.on_serial_full_frame_received(payload)
        if len(host.started) > previous:
            host.complete_current()

    manager.sig_serial_full_frame.connect(process)
    worker.run()
    assert host.started == [c["key"] for c in host.product_test_condition_configs]
    assert bool(port.writes) is polling
    assert all(seconds == 0.001 for seconds in cycles)
    assert not port.is_open


@pytest.mark.parametrize("polling", [False, True])
def test_connection_check_obeys_switch_and_closes_port(monkeypatch, polling):
    class Port:
        def __init__(self):
            self.writes = []
            self.closed = False
            self.data = b"" if polling else bytes.fromhex(A)

        def reset_input_buffer(self):
            pass

        def write(self, data):
            self.writes.append(data)
            self.data = bytes.fromhex(A)

        @property
        def in_waiting(self):
            return len(self.data)

        def read(self, size):
            return self.data

        def close(self):
            self.closed = True

    port = Port()
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=lambda **kwargs: port))
    result = UnifiedHardwareManager().test_serial_discrete_input_connection(configuration(polling))
    assert result["ok"] and result["raw_hex"] == A
    assert port.writes == ([bytes.fromhex(QUERY)] if polling else [])
    assert port.closed


@pytest.mark.parametrize("failure", ["no_response", "write_error"])
def test_connection_failure_closes_port(monkeypatch, failure):
    class Port:
        in_waiting = 0
        closed = False

        def reset_input_buffer(self):
            pass

        def write(self, data):
            if failure == "write_error":
                raise OSError("disconnected")

        def close(self):
            self.closed = True

    port = Port()
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=lambda **kwargs: port))
    result = UnifiedHardwareManager().test_serial_discrete_input_connection(configuration(True))
    assert not result["ok"] and not result["raw_hex"]
    assert port.closed


@pytest.mark.parametrize("polling", [False, True])
def test_product_connection_entry_keeps_polling_settings(polling):
    from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin

    received = []

    def check(config):
        received.append(config)
        return {"ok": True, "raw_hex": A}

    host = SimpleNamespace(hw_manager=SimpleNamespace(
        serial_config={}, serial_worker=None, test_serial_discrete_input_connection=check,
    ))
    config = configuration(polling)
    result = SequenceWidgetSerialTriggerOpsMixin._test_serial_trigger_connection(host, config)
    assert result["connected"] and result["has_response"]
    assert received[0]["polling_settings"] == config["polling_settings"]


@pytest.mark.parametrize("field,value", [("enabled", True), ("query_command_hex", "01 02"), ("interval_ms", 100)])
def test_reloading_polling_changes_restarts_worker(monkeypatch, field, value):
    class Signal:
        def connect(self, callback):
            pass

    class Worker:
        def __init__(self, config, **kwargs):
            self.config = deepcopy(config)
            self.sig_state_changed = Signal()
            self.sig_status = Signal()
            self.running = True

        def isRunning(self):
            return self.running

        def start(self):
            pass

        def stop(self):
            self.running = False

    monkeypatch.setattr(worker_module, "SerialDiscreteInputWorker", Worker)
    manager = UnifiedHardwareManager()
    config = configuration()
    assert manager.start_serial_discrete_input_listener(config, [A])["ok"]
    first = manager.serial_worker
    assert manager.start_serial_discrete_input_listener(config, [A])["message"] == "already running"
    assert manager.serial_worker is first
    modified = deepcopy(config)
    modified["polling_settings"][field] = value
    assert manager.start_serial_discrete_input_listener(modified, [A])["ok"]
    assert manager.serial_worker is not first and not first.running
    assert manager.serial_worker.config == modified
    invalid = deepcopy(modified)
    invalid["polling_settings"]["interval_ms"] = 0
    current = manager.serial_worker
    assert not manager.start_serial_discrete_input_listener(invalid, [A])["ok"]
    assert manager.serial_worker is current and current.running
    manager.stop_serial_discrete_input_listener()


def test_save_removes_legacy_fields_preserves_sequence_and_validates(tmp_path, monkeypatch):
    path = tmp_path / "serial.json"
    config = configuration(True)
    config["state_maps"] = {"full_frame": {A: {"action": "start_record"}}}
    config["trigger_settings"] = {"delay_seconds": 0.5}
    config["decoder"] = {"mode": "state_byte", "state_byte_index": 3}
    config["_comment_decoder"] = "obsolete"
    config["simulator_sequence"] = {"steps": [{"hex": A}]}
    config["serial_settings"]["_comment"] = "old port comment"
    config["polling_settings"]["_comment_enabled"] = "old mode comment"
    config["simulator_sequence"]["steps"][0]["_comment"] = "old step comment"
    input_snapshot = deepcopy(config)
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(path))
    assert "_comment" not in json.dumps(LoadUiConfig.load_serial_discrete_input_config()[1])
    assert LoadUiConfig.save_serial_discrete_input_config(config)
    saved = json.loads(path.read_text(encoding="utf-8"))
    defaults = LoadUiConfig.get_default_serial_discrete_input_config()
    for key in ("state_maps", "trigger_settings", "decoder", "_comment_decoder"):
        assert key not in saved
        assert key not in defaults
    assert saved["simulator_sequence"] == {"steps": [{"hex": A}]}
    assert "_comment" not in json.dumps(saved)
    assert config == input_snapshot
    assert LoadUiConfig.load_serial_discrete_input_config()[1] == saved
    original = path.read_bytes()
    config["polling_settings"]["query_command_hex"] = ""
    assert not LoadUiConfig.save_serial_discrete_input_config(config)
    assert path.read_bytes() == original


@pytest.mark.parametrize("retry_wait", [False, True])
def test_stop_wakes_real_thread_wait_without_waiting_full_interval(monkeypatch, retry_wait):
    monkeypatch.setattr(worker_module.LogManager, "set_log_handler", lambda name: _Logger())
    config = configuration(True)
    config["polling_settings"]["interval_ms"] = 1200
    ports = []

    class Port:
        is_open = True
        in_waiting = 0

        def __init__(self):
            self.writes = 0
            ports.append(self)

        def reset_input_buffer(self):
            pass

        def write(self, data):
            self.writes += 1
            if retry_wait:
                raise OSError("test transport failure")

        def close(self):
            self.is_open = False

    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **kwargs: Port()))
    # Each listener is a new worker, matching the manager's restart behavior.
    for _ in range(2):
        worker = worker_module.SerialDiscreteInputWorker(config, [A])
        entered_wait = threading.Event()
        wait = worker._stop_event.wait
        durations = []

        def observe_wait(seconds):
            durations.append(seconds)
            entered_wait.set()
            return wait(seconds)

        monkeypatch.setattr(worker._stop_event, "wait", observe_wait)
        worker.start()
        try:
            assert entered_wait.wait(2)
            began = time.monotonic()
            worker.stop()
            elapsed = time.monotonic() - began
            assert not worker.isRunning()
            assert elapsed < 0.75
            assert durations == [1 if retry_wait else 1.2]
            assert ports[-1].writes == 1
            assert not ports[-1].is_open
        finally:
            worker.stop()
            assert worker.wait(3000)


def test_stop_before_run_does_not_get_cleared(monkeypatch):
    monkeypatch.setattr(worker_module.LogManager, "set_log_handler", lambda name: _Logger())
    opened = []
    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **kwargs: opened.append(kwargs)))
    worker = worker_module.SerialDiscreteInputWorker(configuration(True), [A])
    worker.stop()
    worker.run()
    assert opened == []
