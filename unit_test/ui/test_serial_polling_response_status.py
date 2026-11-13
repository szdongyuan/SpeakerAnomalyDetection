"""Exercise the worker clock, manager and existing connection button without hardware."""

from types import SimpleNamespace

import pytest

from base.hardware_trigger import serial_discrete_input_worker as worker_module
from base.load_config import LoadUiConfig
from base.unified_hid_device_manager import UnifiedHardwareManager
import logging

A = "A5 5A 01 01 0D 0A"
B = "A5 5A 01 02 0D 0A"


def run_timeline(monkeypatch, timeline, *, polling=True, product_frames=True, empty_read=False):
    config = LoadUiConfig.normalize_serial_discrete_input_config({
        "enabled": True,
        "serial_settings": {"port": "FAKE"},
        "polling_settings": {"enabled": polling},
    })
    monkeypatch.setattr(worker_module.LogManager, "set_log_handler", logging.getLogger)
    worker = worker_module.SerialDiscreteInputWorker(config, [A, B] if product_frames else None)
    manager = UnifiedHardwareManager()
    manager.serial_config = config
    manager.serial_worker = worker
    worker.sig_status.connect(manager._on_serial_worker_status)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    statuses = []
    manager.sig_serial_trigger_status.connect(lambda status: statuses.append((clock.now, status)))
    snapshots = []
    events = iter(timeline)
    current = None

    class Port:
        is_open = True
        data = b""
        writes = 0

        def reset_input_buffer(self):
            pass

        def write(self, data):
            self.writes += 1
            return len(data)

        @property
        def in_waiting(self):
            return len(self.data) or (1 if empty_read else 0)

        def read(self, size):
            data, self.data = self.data, b""
            return data

        def close(self):
            self.is_open = False

    port = Port()

    def advance_clock(_seconds):
        nonlocal current
        if current is not None:
            snapshots.append((clock.now, "已连接" if manager.get_serial_discrete_input_status()["has_response"] else "已打开", manager.get_serial_discrete_input_status()))
        current = next(events, None)
        if current is None:
            # The test stops observation intentionally, independently of response timeout.
            return True
        clock.now, port.data = current
        return False

    monkeypatch.setattr(worker._stop_event, "wait", advance_clock)
    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **kwargs: port))
    worker.run()
    assert not port.is_open
    assert all(snapshot["connected"] for _, _, snapshot in snapshots)
    assert bool(port.writes) == polling
    return snapshots, [(at, status) for at, status in statuses if status["running"]]


@pytest.mark.parametrize("product_frames", [False, True])
@pytest.mark.parametrize("empty_read", [False, True])
def test_polling_three_second_boundary_recovers_and_rearms(
    ui_qapp, monkeypatch, product_frames, empty_read,
):
    timeline = [
        (0.0, bytes.fromhex(A)), (2.999, b""), (3.0, b""), (4.0, b""),
        (5.0, bytes.fromhex(A)), (7.999, b""), (8.0, b""),
    ]
    snapshots, statuses = run_timeline(
        monkeypatch, timeline, product_frames=product_frames, empty_read=empty_read,
    )
    assert [(at, text) for at, text, _ in snapshots] == [
        (0.0, "已连接"), (2.999, "已连接"), (3.0, "已打开"), (4.0, "已打开"),
        (5.0, "已连接"), (7.999, "已连接"), (8.0, "已打开"),
    ]
    timeouts = [(at, status) for at, status in statuses if "无响应" in status["message"]]
    assert [at for at, _ in timeouts] == [3.0, 8.0]
    assert all(status["connected"] and not status["error"] for _, status in timeouts)


def test_no_initial_reply_stays_open_and_reports_timeout_once(ui_qapp, monkeypatch):
    snapshots, statuses = run_timeline(monkeypatch, [(2.999, b""), (3.0, b""), (10.0, b"")])
    assert all(text == "已打开" for _, text, _ in snapshots)
    assert [at for at, status in statuses if "无响应" in status["message"]] == [3.0]


@pytest.mark.parametrize("product_frames", [False, True])
def test_passive_silence_keeps_last_connection_state(ui_qapp, monkeypatch, product_frames):
    snapshots, statuses = run_timeline(
        monkeypatch, [(3.0, b""), (10.0, bytes.fromhex(A)), (13.0, b""), (100.0, b"")],
        polling=False, product_frames=product_frames,
    )
    assert [(at, text) for at, text, _ in snapshots] == [
        (3.0, "已打开"), (10.0, "已连接"), (13.0, "已连接"), (100.0, "已连接"),
    ]
    assert not any("无响应" in status["message"] for _, status in statuses)


def test_each_received_chunk_resets_deadline(ui_qapp, monkeypatch):
    snapshots, statuses = run_timeline(monkeypatch, [
        (0.0, bytes.fromhex(A)), (2.9, b"\xA5"), (3.0, b""), (5.899, b""), (5.901, b""),
    ])
    assert [text for _, text, _ in snapshots] == ["已连接"] * 4 + ["已打开"]
    assert [at for at, status in statuses if "无响应" in status["message"]] == [5.901]
