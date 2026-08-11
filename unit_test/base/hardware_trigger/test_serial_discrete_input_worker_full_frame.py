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
    events = []
    worker.sig_state_changed.connect(events.append)

    worker.run()

    assert port.write_calls == []
    assert [event["raw_hex"] for event in events] == [FRAME]
    assert events[0]["product_full_frame"] is True


def test_active_report_prints_raw_data_and_matched_state_code(
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

    worker.run()

    output = capsys.readouterr().out
    assert f"收到主动上报原始数据: raw_hex={FRAME}" in output
    assert f"匹配完整状态码: frame={FRAME}" in output


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


def test_unconfigured_active_report_is_printed_without_business_event(
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
    assert f"收到主动上报原始数据: raw_hex={UNCONFIGURED_FRAME}" in output
    assert "匹配完整状态码" not in output


def test_legacy_polling_mode_still_writes_the_configured_query(monkeypatch):
    config = {
        "serial_settings": {"port": "FAKE", "timeout": 0.01},
        "polling_settings": {"interval_ms": 1, "query_command_hex": "01 02"},
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
    full_frames = []
    directions = []
    manager.sig_serial_full_frame.connect(full_frames.append)
    manager.sig_directional_trigger.connect(directions.append)

    manager._on_serial_state_changed(
        {
            "mode": "full_frame",
            "value": FRAME,
            "raw_hex": FRAME,
            "product_full_frame": True,
        }
    )

    assert [event["raw_hex"] for event in full_frames] == [FRAME]
    assert directions == []


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
    assert "未收到主动上报报文" in result["message"]
