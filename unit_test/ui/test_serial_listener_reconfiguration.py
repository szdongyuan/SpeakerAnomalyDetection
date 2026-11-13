"""Generic serial listener reconfiguration keeps generation ownership."""
from copy import deepcopy
from threading import Thread

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QObject, pyqtSignal

from base.hardware_trigger import serial_discrete_input_worker as worker_module
from base.load_config import LoadUiConfig
from base.unified_hid_device_manager import UnifiedHardwareManager

A = 'A5 5A 01 01 0D 0A'
B = 'A5 5A 01 02 0D 0A'


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
        self.sig_status.emit({'running': False, 'connected': False, 'message': 'listener stopped'})
    def opened(self):
        self.sig_status.emit({'running': True, 'connected': True, 'error': '', 'message': 'opened'})
    def failed(self):
        self.running = False
        self.sig_status.emit({'running': False, 'connected': False, 'error': 'port unavailable'})
    def frame(self, frame):
        self.sig_state_changed.emit({'raw_hex': frame, 'product_full_frame': True})


@pytest.fixture
def listener(ui_qapp, monkeypatch):
    monkeypatch.setattr(worker_module, 'SerialDiscreteInputWorker', FakeWorker)
    config = LoadUiConfig.normalize_serial_discrete_input_config({
        'enabled': True, 'serial_settings': {'port': 'FAKE'}})
    manager = UnifiedHardwareManager()
    received = []
    manager.sig_serial_full_frame.connect(received.append)
    assert manager.start_serial_discrete_input_listener(config, (A, B))['ok']
    manager.serial_worker.opened()
    yield manager, config, received
    manager.stop_serial_discrete_input_listener()
    ui_qapp.processEvents()


@pytest.mark.parametrize('section,field,value', [
    ('polling_settings', 'enabled', True), ('polling_settings', 'interval_ms', 1200),
    ('serial_settings', 'bytesize', 7), ('serial_settings', 'parity', 'E'),
    ('serial_settings', 'stopbits', 2), ('serial_settings', 'timeout', .3),
])
def test_serial_parameters_restart_with_new_values(listener, section, field, value):
    manager, initial, received = listener
    old = manager.serial_worker
    changed = deepcopy(initial)
    changed[section][field] = value
    assert manager.start_serial_discrete_input_listener(changed, (A, B))['ok']
    assert manager.serial_worker is not old and not old.isRunning()
    assert manager.serial_worker.config[section][field] == value
    manager.serial_worker.opened()
    manager.serial_worker.frame(B)
    assert [event['raw_hex'] for event in received] == [B]


@pytest.mark.parametrize('old_running', [False, True])
@pytest.mark.parametrize('delete_old', [False, True])
def test_queued_old_worker_frames_and_status_cannot_touch_new_listener(listener, ui_qapp, old_running, delete_old):
    manager, initial, received = listener
    old = manager.serial_worker
    def delayed():
        old.frame(B)
        old.sig_status.emit({'running': False, 'connected': False, 'error': 'old worker closed'})
        old.running = old_running
    emitter = Thread(target=delayed)
    emitter.start()
    emitter.join(2)
    assert not emitter.is_alive()
    changed = deepcopy(initial)
    changed['polling_settings']['enabled'] = True
    assert manager.start_serial_discrete_input_listener(changed, (A, B))['ok']
    manager.serial_worker.opened()
    if delete_old:
        sip.delete(old)
    ui_qapp.processEvents()
    assert not received
    assert manager.get_serial_discrete_input_status()['connected']


def test_restart_open_failure_can_retry_without_old_worker(listener):
    manager, config, received = listener
    changed = deepcopy(config)
    changed['polling_settings']['enabled'] = True
    assert manager.start_serial_discrete_input_listener(changed, (A, B))['ok']
    failed = manager.serial_worker
    failed.failed()
    status = manager.get_serial_discrete_input_status()
    assert status['error'] == 'port unavailable' and not status['connected']
    assert manager.start_serial_discrete_input_listener(changed, (A, B))['ok']
    assert manager.serial_worker is not failed
    manager.serial_worker.opened()
    manager.serial_worker.frame(B)
    assert [event['raw_hex'] for event in received] == [B]
