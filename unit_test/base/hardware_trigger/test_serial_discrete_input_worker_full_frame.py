import sys
from types import SimpleNamespace

import pytest

from base.hardware_trigger import serial_discrete_input_worker as worker_module
from base.hardware_trigger.serial_discrete_input_worker import SerialDiscreteInputWorker
from base.unified_hid_device_manager import UnifiedHardwareManager


FRAME = "01 04 02 00 01 78 F0"
UNCONFIGURED_FRAME = "01 04 02 00 02 38 F1"


class _Logger:
    def __init__(self):
        self.infos = []

    def info(self, message):
        self.infos.append(message)

    def error(self, _message):
        return None


@pytest.fixture(autouse=True)
def _isolate_log_manager(monkeypatch):
    monkeypatch.setattr(
        worker_module.LogManager,
        "set_log_handler",
        staticmethod(lambda _name: _Logger()),
    )


class _FakeSerialPort:
    def __init__(self, worker, data):
        self.worker = worker
        self.data = bytes(data)
        self.is_open = True
        self.write_calls = []

    @property
    def in_waiting(self):
        return len(self.data)

    def reset_input_buffer(self):
        return None

    def write(self, data):
        self.write_calls.append(bytes(data))

    def read(self, _size):
        data = self.data
        self.data = b""
        self.worker._is_running = False
        return data

    def close(self):
        self.is_open = False


def test_active_report_full_frame_mode_does_not_write_query(monkeypatch):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {
            "interval_ms": 1,
            "query_command_hex": "NOT USED IN ACTIVE REPORT MODE",
        },
        "decoder": {"mode": "state_byte", "state_byte_index": 3},
    }
    worker = SerialDiscreteInputWorker(config, full_frame_candidates=[FRAME])
    port = _FakeSerialPort(worker, bytes.fromhex(FRAME))
    monkeypatch.setattr(
        worker_module,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: port),
    )
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: 123.5)
    monkeypatch.setattr(worker_module, "perf_counter_ns", lambda: 123456789)
    events = []
    worker.sig_state_changed.connect(events.append)

    worker.run()

    assert port.write_calls == []
    assert [event["raw_hex"] for event in events] == [FRAME]
    assert events[0]["product_full_frame"] is True
    assert events[0]["received_monotonic"] == 123.5
    assert events[0]["received_perf_ns"] == 123456789


def test_active_report_logs_raw_data_and_matched_state_code_without_stdout(
    monkeypatch,
    capsys,
):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {"interval_ms": 1, "query_command_hex": ""},
    }
    worker = SerialDiscreteInputWorker(
        config,
        full_frame_candidates=[FRAME],
    )
    port = _FakeSerialPort(worker, bytes.fromhex(FRAME))
    monkeypatch.setattr(
        worker_module,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: port),
    )
    worker.logger = _Logger()

    worker.run()

    output = capsys.readouterr().out
    assert output == ""
    assert f"serial_product_raw_received raw_hex={FRAME}" in worker.logger.infos
    assert f"serial_product_frame_matched frame={FRAME}" in worker.logger.infos


def test_active_report_worker_emits_every_transport_duplicate(monkeypatch):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {"interval_ms": 1, "query_command_hex": ""},
    }
    worker = SerialDiscreteInputWorker(config, full_frame_candidates=[FRAME])
    port = _FakeSerialPort(worker, bytes.fromhex(FRAME + " " + FRAME))
    monkeypatch.setattr(
        worker_module,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: port),
    )
    events = []
    worker.sig_state_changed.connect(events.append)

    worker.run()

    assert [event["raw_hex"] for event in events] == [FRAME, FRAME]


def test_delayed_match_keeps_the_read_time_of_each_frames_last_byte(monkeypatch):
    short = "00 00 00 00"
    long = "A5 00 00 00 00 0D"
    worker = SerialDiscreteInputWorker(
        {"serial_settings": {"port": "FAKE"}}, full_frame_candidates=[short, long],
    )
    chunks = [bytes.fromhex(part) for part in ("A5 00", "00 00 00", "FF 00 00", "00 00")]
    port = _FakeSerialPort(worker, chunks[0])
    read_count = 0
    events = []

    def read(_size):
        nonlocal read_count
        chunk = chunks.pop(0)
        read_count += 1
        port.data = chunks[0] if chunks else b""
        worker._is_running = bool(chunks)
        return chunk

    monkeypatch.setattr(port, "read", read)
    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **kwargs: port))
    monkeypatch.setattr(worker_module, "perf_counter_ns", lambda: read_count * 100)
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: float(read_count)))
    worker.sig_state_changed.connect(lambda event: events.append((
        read_count, event["raw_hex"], event["received_perf_ns"], event["received_monotonic"],
    )))
    worker.run()

    # First B was complete on read 2, but the long candidate is ruled out only on read 3.
    assert events == [(3, short, 200, 2.0), (4, short, 400, 4.0)]


def test_unconfigured_active_report_is_logged_without_stdout_or_business_event(
    monkeypatch,
    capsys,
):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {"interval_ms": 1, "query_command_hex": ""},
    }
    worker = SerialDiscreteInputWorker(config, full_frame_candidates=[FRAME])
    worker.logger = _Logger()
    port = _FakeSerialPort(worker, bytes.fromhex(UNCONFIGURED_FRAME))
    monkeypatch.setattr(
        worker_module,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: port),
    )
    events = []
    worker.sig_state_changed.connect(events.append)

    worker.run()

    assert events == []
    assert any(
        f"serial_product_raw_received raw_hex={UNCONFIGURED_FRAME}" in message
        for message in worker.logger.infos
    )
    output = capsys.readouterr().out
    assert output == ""
    assert not any("serial_product_frame_matched" in message for message in worker.logger.infos)


def test_legacy_polling_mode_writes_query_when_explicitly_enabled(monkeypatch):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {"enabled": True, "interval_ms": 1, "query_command_hex": "01 02"},
        "decoder": {"mode": "full_frame"},
    }
    worker = SerialDiscreteInputWorker(config)
    port = _FakeSerialPort(worker, bytes.fromhex(FRAME))
    monkeypatch.setattr(
        worker_module,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: port),
    )

    worker.run()

    assert port.write_calls == [b"\x01\x02"]


def test_hardware_manager_forwards_product_full_frame_without_direction_mapping():
    manager = UnifiedHardwareManager()
    manager.serial_config = {"enabled": True}
    worker = SerialDiscreteInputWorker(manager.serial_config, [FRAME])
    manager.serial_worker = worker
    worker.sig_state_changed.connect(manager._on_serial_state_changed)
    full_frames = []
    directions = []
    manager.sig_serial_full_frame.connect(full_frames.append)
    manager.sig_directional_trigger.connect(directions.append)

    worker.sig_state_changed.emit(
        {
            "mode": "full_frame",
            "value": FRAME,
            "raw_hex": FRAME,
            "product_full_frame": True,
            "received_monotonic": 123.5,
            "received_perf_ns": 123456789,
        }
    )

    assert [event["raw_hex"] for event in full_frames] == [FRAME]
    assert directions == []
    assert full_frames[0]["received_monotonic"] == 123.5
    assert full_frames[0]["received_perf_ns"] == 123456789


def test_passive_connection_test_treats_an_open_port_as_connected(monkeypatch):
    class _PassivePort:
        in_waiting = 0

        def reset_input_buffer(self):
            return None

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "serial",
        SimpleNamespace(Serial=lambda **_kwargs: _PassivePort()),
    )
    manager = UnifiedHardwareManager()

    result = manager.test_serial_discrete_input_connection(
        {
            "serial_settings": {"port": "FAKE", "timeout": 0.01},
            "polling_settings": {"query_command_hex": ""},
        }
    )

    assert result["ok"] is True
    assert result["raw_hex"] == ""
    assert result["message"] == "串口测试通过，暂未收到数据。"
