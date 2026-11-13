"""Application exit drains the shared CSV owner while Qt remains responsive."""
import multiprocessing
from time import monotonic
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtTest import QTest

from base.raw_audio_csv_service import RawAudioCsvService
from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
from unit_test.base.test_raw_audio_csv_worker import make_command
from unit_test.ui.test_workbench_recording_startup import startup


def pump(app, predicate, timeout=15):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        app.processEvents()
        QTest.qWait(2)
    assert predicate()


@pytest.fixture(params=['zip_write', 'zip_verify'])
def runtime(ui_qapp, request):
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    context = multiprocessing.get_context('spawn')
    entered, gate = context.Event(), context.Event()
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(request.param, entered, gate))
    service.test_zip_entered = entered
    bridge = RawAudioCsvServiceBridge(service)
    yield service, bridge, gate
    gate.set()
    for token in tuple(service._ledger._reservations.values()):
        service.release_reservation(token)
    service.begin_shutdown()
    pump(ui_qapp, service.closed.is_set)
    bridge.close_delivery()


def submit(service, tmp_path, name='one'):
    request = make_command(tmp_path, name).request
    token = service.reserve(request.recording_id).reservation
    assert service.commit(token, request) == 'accepted'
    return request


def test_blocked_export_close_keeps_qt_alive_and_drains(startup, ui_qapp, runtime, tmp_path):
    service, bridge, gate = runtime
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    submit(service, tmp_path)
    pump(ui_qapp, service.test_zip_entered.is_set)
    pid = service._process.pid
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        for _ in range(3):
            window.close()
            ui_qapp.processEvents()
            assert not getattr(window, '_exit_cleanup_complete', False)
        assert '保存/压缩原始 CSV' in window.statusBar().currentMessage()
        pump(ui_qapp, lambda: len(ticks) >= 5)
        assert window.isVisible()
        assert service._process.pid == pid
        assert service.snapshot().outstanding == 1
        assert not window.sequence_window._workbench_input_closed
        gate.set()
        pump(ui_qapp, lambda: getattr(window, '_exit_cleanup_complete', False))
        assert service.closed.is_set()
        assert not service._thread.is_alive()
        assert service._control is service._process is None
        assert pid not in [child.pid for child in multiprocessing.active_children()]
        assert window.sequence_window._workbench_input_closed
        assert startup.bridge.shutdown_calls == 1
        assert startup.analysis_service.shutdown_calls == 1
    finally:
        timer.stop()
        gate.set()


class Video(QObject):
    closed = pyqtSignal()
    is_shutdown_complete = False
    def __init__(self):
        super().__init__()
        self.shutdown = Mock()


def test_main_closes_admission_before_video_wait(startup, ui_qapp, runtime):
    service, bridge, _ = runtime
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    original = window.video_controller
    window.video_controller = Video()
    try:
        window.close()
        window.close()
        assert window._closing
        assert service.reserve('new').status == 'closing'
        assert not window._workbench_scanner_admission_available()
        assert not window._calibration_admission_available()
        window.video_controller.shutdown.assert_called_once()
        assert not window.sequence_window._workbench_input_closed
        window.video_controller.is_shutdown_complete = True
        window.video_controller.closed.emit()
        pump(ui_qapp, lambda: getattr(window, '_exit_cleanup_complete', False))
    finally:
        original.shutdown()


@pytest.mark.parametrize('late_commit', [False, True])
def test_recording_reservation_resolves_during_main_drain(startup, ui_qapp, runtime, tmp_path, late_commit):
    service, bridge, gate = runtime
    request = make_command(tmp_path, 'late').request
    token = service.reserve(request.recording_id).reservation
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    try:
        window.close()
        assert not getattr(window, '_exit_cleanup_complete', False)
        assert service.snapshot().phase == 'draining'
        assert service.reserve('new').status == 'closing'
        if late_commit:
            assert service.commit(token, request) == 'accepted'
        else:
            assert service.release_reservation(token)
        gate.set()
        pump(ui_qapp, lambda: getattr(window, '_exit_cleanup_complete', False))
        assert service.snapshot().outstanding == 0
        assert window.sequence_window._workbench_input_closed
    finally:
        gate.set()


def test_export_failure_still_closes_after_resource_release(startup, ui_qapp, tmp_path):
    from pathlib import Path
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    events = []
    bridge.subscribe(events.append)
    request = make_command(tmp_path, 'missing').request
    Path(request.wav_path).unlink()
    service.commit(service.reserve(request.recording_id).reservation, request)
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    try:
        window.close()
        pump(ui_qapp, lambda: getattr(window, '_exit_cleanup_complete', False))
        terminal, = [event for event in events if event.kind == 'terminal']
        assert terminal.result.stage == 'inspect'
        assert service.snapshot().outstanding == 0
        assert bridge.service_closed
        assert not service._thread.is_alive()
        assert service._process is service._control is None
    finally:
        service.begin_shutdown()
        assert service.closed.wait(15)
        bridge.close_delivery()


def test_incomplete_recording_shutdown_does_not_fabricate_csv_release(startup, ui_qapp, runtime, tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    from PyQt5.QtWidgets import QMessageBox
    service, bridge, _ = runtime
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    recording = window.recording_bridge
    callbacks = []
    closed = Event()
    path = str(tmp_path / 'unreleased.wav')
    permit = service.try_acquire_mutation((path,))
    window.recording_bridge = SimpleNamespace(
        service=SimpleNamespace(closed=closed, diagnostics=['reader still owns WAV']),
        shutdown=callbacks.append)
    warning = Mock()
    monkeypatch.setattr(QMessageBox, 'warning', warning)
    try:
        window.close()
        callbacks[0]()
        assert window.isVisible()
        assert not closed.is_set()
        assert not bridge.service_closed
        assert service.paths_busy((path,))
        assert not window.sequence_window._workbench_input_closed
        warning.assert_called_once()
        service.release_mutation(permit)
        pump(ui_qapp, lambda: getattr(window, '_exit_cleanup_complete', False))
    finally:
        service.release_mutation(permit)
        recording.shutdown()
